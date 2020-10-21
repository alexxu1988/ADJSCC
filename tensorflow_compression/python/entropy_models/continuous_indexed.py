# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Indexed entropy model for continuous random variables."""

import tensorflow.compat.v2 as tf

from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.entropy_models import continuous_base
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import range_coding_ops


__all__ = [
    "ContinuousIndexedEntropyModel",
    "LocationScaleIndexedEntropyModel",
]


class ContinuousIndexedEntropyModel(continuous_base.ContinuousEntropyModelBase):
  """Indexed entropy model for continuous random variables.

  This entropy model handles quantization of a bottleneck tensor and helps with
  training of the parameters of the probability distribution modeling the
  tensor. It also pre-computes integer probability tables, which can then be
  used to compress and decompress bottleneck tensors reliably across different
  platforms.

  A typical workflow looks like this:

  - Train a model using this entropy model as a bottleneck, passing the
    bottleneck tensor through `quantize()` while optimizing compressibility of
    the tensor using `bits()`. `bits(training=True)` computes a differentiable
    upper bound on the number of bits needed to compress the bottleneck tensor.
  - For evaluation, get a closer estimate of the number of compressed bits
    using `bits(training=False)`.
  - Call `update_tables()` to ensure the probability tables for range coding are
    up-to-date.
  - Share the model between a sender and a receiver.
  - On the sender side, compute the bottleneck tensor and call `compress()` on
    it. The output is a compressed string representation of the tensor. Transmit
    the string to the receiver, and call `decompress()` there. The output is the
    quantized bottleneck tensor. Continue processing the tensor on the receiving
    side.

  This class assumes that all scalar elements of the encoded tensor are
  conditionally independent given some other random variable, possibly depending
  on data. All dependencies must be represented by the `indexes` tensor. For
  each bottleneck tensor element, it selects the appropriate scalar
  distribution.

  The `indexes` tensor must contain only integer values (but may have
  floating-point type for purposes of backpropagation) in a pre-specified range.
  If `index_ranges` is a single integer, the index values must be in the range
  `[0, index_ranges)` and `indexes` must have the same shape as the bottleneck
  tensor. This only allows a one-dimensional conditional dependency. To make the
  distribution conditionally dependent on `n`-dimensional indexes,
  `index_ranges` must be specified as an iterable of `n` integers. Then,
  `indexes` must have the same shape as the bottleneck tensor with an additional
  channel dimension of length `n`. The position of the channel dimension is
  given by `channel_axis`. The index values in the `n`th channel must be in the
  range `[0, index_ranges[n])`.

  The implied distribution for the bottleneck tensor is determined as:
  ```
  distribution_fn(**{k: f(indexes) for k, f in parameter_fns.items()})
  ```

  A more detailed description (and motivation) of this indexing scheme can be
  found in the following paper. Please cite the paper when using this code for
  derivative work.

  > "Integer Networks for Data Compression with Latent-Variable Models"<br />
  > J. Ballé, N. Johnston, D. Minnen<br />
  > https://openreview.net/forum?id=S1zz2i0cY7

  Examples:

  To make a parameterized zero-mean normal distribution, one could use:
  ```
  tfc.ContinuousIndexedEntropyModel(
      distribution_fn=tfc.NoisyNormal,
      index_ranges=64,
      parameter_fns=dict(
          loc=lambda _: 0.,
          scale=lambda i: tf.exp(i / 8 - 5),
      ),
      coding_rank=1,
  )
  ```
  Then, each element of `indexes` in the range `[0, 64)` would indicate that the
  corresponding element in `bottleneck` is normally distributed with zero mean
  and a standard deviation between `exp(-5)` and `exp(2.875)`, inclusive.

  To make a parameterized logistic mixture distribution, one could use:
  ```
  tfc.ContinuousIndexedEntropyModel(
      distribution_fn=tfc.NoisyLogisticMixture,
      index_ranges=(10, 10, 5),
      parameter_fns=dict(
          loc=lambda i: i[..., 0:2] - 5,
          scale=lambda _: 1,
          weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
      ),
      coding_rank=1,
      channel_axis=-1,
  )
  ```
  Then, the last dimension of `indexes` would consist of triples of elements in
  the ranges `[0, 10)`, `[0, 10)`, and `[0, 5)`, respectively. Each triples
  would indicate that the element in `bottleneck` corresponding to the other
  dimensions is distributed with a mixture of two logistic distributions, where
  the components each have one of 10 location parameters between `-5` and `+5`,
  inclusive, unit scale parameters, and one of five different mixture
  weightings.
  """

  def __init__(self, distribution_fn, index_ranges, parameter_fns, coding_rank,
               channel_axis=-1, dtype=tf.float32, likelihood_bound=1e-9,
               tail_mass=2**-8, range_coder_precision=12):
    """Initializer.

    Arguments:
      distribution_fn: A callable returning a `tfp.distributions.Distribution`
        object, which is used to model the distribution of the bottleneck tensor
        values including additive uniform noise - typically a `Distribution`
        class or factory function. The callable will receive keyword arguments
        as determined by `parameter_fns`. For best results, the distributions
        should be flexible enough to have a unit-width uniform distribution as a
        special case, since this is the distribution an element will take on
        when its bottleneck value is constant (due to the additive noise).
      index_ranges: Integer or iterable of integers. If a single integer,
        `indexes` must have the same shape as `bottleneck`, and `channel_axis`
        is ignored. Its values must be in the range `[0, index_ranges)`. If an
        iterable of integers, `indexes` must have an additional dimension at
        position `channel_axis`, and the values of the `n`th channel must be in
        the range `[0, index_ranges[n])`.
      parameter_fns: Dict of strings to callables. Functions mapping `indexes`
        to each distribution parameter. For each item, `indexes` is passed to
        the callable, and the string key and return value make up one keyword
        argument to `distribution_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        `bits()` method sums over each coding unit.
      channel_axis: Integer. For iterable `index_ranges`, determines the
        position of the channel axis in `indexes`. Defaults to the last
        dimension.
      dtype: `tf.dtypes.DType`. The data type of all floating-point
        computations carried out in this class.
      likelihood_bound: Float. Lower bound for likelihood values, to prevent
        training instabilities.
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
    """
    if coding_rank <= 0:
      raise ValueError("`coding_rank` must be larger than 0.")

    self._distribution_fn = distribution_fn
    if not callable(self.distribution_fn):
      raise TypeError("`distribution_fn` must be a class or factory function.")
    try:
      self._index_ranges = int(index_ranges)
    except TypeError:
      self._index_ranges = tuple(int(r) for r in index_ranges)  # pytype: disable=attribute-error
    self._parameter_fns = dict(parameter_fns)
    for name, fn in self.parameter_fns.items():
      if not isinstance(name, str):
        raise TypeError("`parameter_fns` must have string keys.")
      if not callable(fn):
        raise TypeError("`parameter_fns['{}']` must be callable.".format(name))
    self._channel_axis = int(channel_axis)
    dtype = tf.as_dtype(dtype)

    if isinstance(self.index_ranges, int):
      indexes = tf.range(self.index_ranges, dtype=dtype)
    else:
      indexes = [tf.range(r, dtype=dtype) for r in self.index_ranges]
      indexes = tf.meshgrid(*indexes, indexing="ij")
      indexes = tf.stack(indexes, axis=self.channel_axis)
    parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
    distribution = self.distribution_fn(**parameters)  # pylint:disable=not-callable
    tf.print(distribution.batch_shape)
    tf.print(distribution.event_shape)

    super().__init__(
        distribution, coding_rank, likelihood_bound=likelihood_bound,
        tail_mass=tail_mass, range_coder_precision=range_coder_precision)

  @property
  def index_ranges(self):
    """Upper bound(s) on values allowed in `indexes` tensor."""
    return self._index_ranges

  @property
  def parameter_fns(self):
    """Functions mapping `indexes` to each distribution parameter."""
    return self._parameter_fns

  @property
  def distribution_fn(self):
    """Class or factory function returning a `Distribution` object."""
    return self._distribution_fn

  @property
  def channel_axis(self):
    """Position of channel axis in `indexes` tensor."""
    return self._channel_axis

  def _make_distribution(self, indexes):
    indexes = tf.cast(indexes, self.dtype)
    parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
    return self.distribution_fn(**parameters)  # pylint:disable=not-callable

  def _normalize_indexes(self, indexes):
    indexes = math_ops.lower_bound(indexes, 0)
    if isinstance(self.index_ranges, int):
      indexes = math_ops.upper_bound(indexes, self.index_ranges - 1)
    else:
      axes = [1] * indexes.shape.rank
      axes[self.channel_axis] = len(self.index_ranges)
      bounds = tf.reshape([s - 1 for s in self.index_ranges], axes)
      indexes = math_ops.upper_bound(indexes, bounds)
    return indexes

  def _flatten_indexes(self, indexes):
    indexes = tf.cast(indexes, tf.int32)
    if isinstance(self.index_ranges, int):
      return indexes
    else:
      strides = tf.cumprod(self.index_ranges, exclusive=True, reverse=True)
      return tf.linalg.tensordot(indexes, strides, [[self.channel_axis], [0]])

  def bits(self, bottleneck, indexes, training=True):
    """Estimates the number of bits needed to compress a tensor.

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution computed by `self.distribution_fn`,
        which is a non-differentiable, tight *lower* bound on the number of bits
        needed to compress `bottleneck` using `compress()`. If `True`, returns a
        somewhat looser, but differentiable *upper* bound on this quantity.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing the number of bits.
    """
    indexes = self._normalize_indexes(indexes)
    distribution = self._make_distribution(indexes)
    if training:
      quantized = bottleneck + tf.random.uniform(
          tf.shape(bottleneck), minval=-.5, maxval=.5, dtype=bottleneck.dtype)
    else:
      offset = helpers.quantization_offset(distribution)
      quantized = self._quantize(bottleneck, offset)
    probs = distribution.prob(quantized)
    probs = math_ops.lower_bound(probs, self.likelihood_bound)
    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(tf.math.log(probs), axis=axes) / -tf.math.log(2.)
    return bits

  def quantize(self, bottleneck, indexes):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values
    modulo a quantization offset, which depends on `indexes`. For instance, for
    Gaussian distributions, the returned values are rounded to the location of
    the mode of the distributions plus or minus an integer.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be quantized.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    indexes = self._normalize_indexes(indexes)
    offset = helpers.quantization_offset(self._make_distribution(indexes))
    return self._quantize(bottleneck, offset)

  def compress(self, bottleneck, indexes):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `indexes`. The quantized tensor can later be recovered by calling
    `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)

    symbols_shape = tf.shape(flat_indexes)
    batch_shape = symbols_shape[:-self.coding_rank]
    flat_shape = tf.concat([[-1], symbols_shape[-self.coding_rank:]], 0)

    flat_indexes = tf.reshape(flat_indexes, flat_shape)

    offset = helpers.quantization_offset(self._make_distribution(indexes))
    symbols = tf.cast(tf.round(bottleneck - offset), tf.int32)
    symbols = tf.reshape(symbols, flat_shape)

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      def loop_body(args):
        return range_coding_ops.unbounded_index_range_encode(
            args[0], args[1],
            self._cdf, self._cdf_length, self._cdf_offset,
            precision=self.range_coder_precision,
            overflow_width=4, debug_level=1)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      strings = tf.map_fn(
          loop_body, (symbols, flat_indexes), dtype=tf.string, name="compress")

    strings = tf.reshape(strings, batch_shape)
    return strings

  def decompress(self, strings, indexes):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Arguments:
      strings: `tf.Tensor` containing the compressed bit strings.
      indexes: `tf.Tensor` specifying the scalar distribution for each output
        element. See class docstring for examples.

    Returns:
      A `tf.Tensor` of the same shape as `indexes` (without the optional channel
      dimension).
    """
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)

    symbols_shape = tf.shape(flat_indexes)
    flat_shape = tf.concat([[-1], symbols_shape[-self.coding_rank:]], 0)

    flat_indexes = tf.reshape(flat_indexes, flat_shape)

    strings = tf.reshape(strings, [-1])

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      def loop_body(args):
        return range_coding_ops.unbounded_index_range_decode(
            args[0], args[1],
            self._cdf, self._cdf_length, self._cdf_offset,
            precision=self.range_coder_precision,
            overflow_width=4, debug_level=1)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      symbols = tf.map_fn(
          loop_body, (strings, flat_indexes), dtype=tf.int32, name="decompress")

    symbols = tf.reshape(symbols, symbols_shape)
    offset = helpers.quantization_offset(self._make_distribution(indexes))
    return tf.cast(symbols, self.dtype) + offset


class LocationScaleIndexedEntropyModel(ContinuousIndexedEntropyModel):
  """Indexed entropy model for location-scale family of random variables.

  This class is a common special case of `ContinuousIndexedEntropyModel`. The
  specified distribution is parameterized with `num_scales` values of scale
  parameters. An element-wise location parameter is handled by shifting the
  distributions to zero. Note: this only works for shift-invariant
  distributions, where the `loc` parameter really denotes a translation (i.e.,
  not for the log-normal distribution).
  """

  def __init__(self, distribution_fn, num_scales, scale_fn, coding_rank,
               dtype=tf.float32, likelihood_bound=1e-9, tail_mass=2**-8,
               range_coder_precision=12):
    """Initializer.

    Arguments:
      distribution_fn: A callable returning a `tfp.distributions.Distribution`
        object, which is used to model the distribution of the bottleneck tensor
        values including additive uniform noise - typically a `Distribution`
        class or factory function. The callable will receive a `scale` keyword
        argument as determined by `scale_fn`. For best results, the
        distributions should be flexible enough to have a unit-width uniform
        distribution as a special case, since this is the distribution an
        element will take on when its bottleneck value is constant (due to the
        additive noise).
      num_scales: Integer. Values in `indexes` must be in the range
        `[0, num_scales)`.
      scale_fn: Callable. `indexes` is passed to the callable, and the return
        value is given as `scale` keyword argument to `distribution_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        `bits()` method sums over each coding unit.
      dtype: `tf.dtypes.DType`. The data type of all floating-point
        computations carried out in this class.
      likelihood_bound: Float. Lower bound for likelihood values, to prevent
        training instabilities.
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
    """
    num_scales = int(num_scales)
    super().__init__(
        distribution_fn=distribution_fn,
        index_ranges=num_scales,
        parameter_fns=dict(
            loc=lambda _: 0.,
            scale=scale_fn,
        ),
        coding_rank=coding_rank,
        dtype=dtype,
        likelihood_bound=likelihood_bound,
        tail_mass=tail_mass,
        range_coder_precision=range_coder_precision,
    )

  def bits(self, bottleneck, scale_indexes, loc=None, training=True):
    """Estimates the number of bits needed to compress a tensor.

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution computed by `self.distribution_fn`,
        which is a non-differentiable, tight *lower* bound on the number of bits
        needed to compress `bottleneck` using `compress()`. If `True`, returns a
        somewhat looser, but differentiable *upper* bound on this quantity.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing the number of bits.
    """
    if loc is not None:
      bottleneck -= loc
    return super().bits(bottleneck, scale_indexes, training=training)

  def quantize(self, bottleneck, scale_indexes, loc=None):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values
    modulo a quantization offset, which depends on `indexes`. For instance, for
    Gaussian distributions, the returned values are rounded to the location of
    the mode of the distributions plus or minus an integer.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be quantized.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    if loc is None:
      return super().quantize(bottleneck, scale_indexes)
    else:
      return super().quantize(bottleneck - loc, scale_indexes) + loc

  def compress(self, bottleneck, scale_indexes, loc=None):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `indexes`. The quantized tensor can later be recovered by calling
    `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    if loc is not None:
      bottleneck -= loc
    return super().compress(bottleneck, scale_indexes)

  def decompress(self, strings, scale_indexes, loc=None):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Arguments:
      strings: `tf.Tensor` containing the compressed bit strings.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each output
        element.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        output elements is assumed to be zero. Otherwise, specifies the location
        parameter for each output element. Must have the same shape as
        `scale_indexes`.

    Returns:
      A `tf.Tensor` of the same shape as `scale_indexes`.
    """
    values = super().decompress(strings, scale_indexes)
    if loc is not None:
      values += loc
    return values
