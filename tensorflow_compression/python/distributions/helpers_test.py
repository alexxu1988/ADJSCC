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
"""Tests of uniform noise adapter distribution."""

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_compression.python.distributions import helpers


class HelpersTest(tf.test.TestCase):

  def test_cauchy_quantizes_to_mode(self):
    dist = tfp.distributions.Cauchy(loc=1.5, scale=3.)
    self.assertEqual(helpers.quantization_offset(dist), 1.5)

  def test_gamma_quantizes_to_mode(self):
    dist = tfp.distributions.Gamma(concentration=5., rate=1.)
    self.assertEqual(helpers.quantization_offset(dist), 4.)

  def test_laplace_quantizes_to_mode(self):
    dist = tfp.distributions.Laplace(loc=-2., scale=5.)
    self.assertEqual(helpers.quantization_offset(dist), -2.)

  def test_logistic_quantizes_to_mode(self):
    dist = tfp.distributions.Logistic(loc=-3., scale=1.)
    self.assertEqual(helpers.quantization_offset(dist), -3.)

  def test_lognormal_quantizes_to_mode(self):
    dist = tfp.distributions.LogNormal(loc=4., scale=1.)
    self.assertEqual(helpers.quantization_offset(dist), tf.exp(3.))

  def test_normal_quantizes_to_mode(self):
    dist = tfp.distributions.Normal(loc=3., scale=5.)
    self.assertEqual(helpers.quantization_offset(dist), 3.)

  def test_cauchy_tails_are_in_order(self):
    dist = tfp.distributions.Cauchy(loc=1.5, scale=3.)
    self.assertGreater(
        helpers.upper_tail(dist, 2**-8), helpers.lower_tail(dist, 2**-8))

  def test_laplace_tails_are_in_order(self):
    dist = tfp.distributions.Laplace(loc=-2., scale=5.)
    self.assertGreater(
        helpers.upper_tail(dist, 2**-8), helpers.lower_tail(dist, 2**-8))

  def test_logistic_tails_are_in_order(self):
    dist = tfp.distributions.Logistic(loc=-3., scale=1.)
    self.assertGreater(
        helpers.upper_tail(dist, 2**-8), helpers.lower_tail(dist, 2**-8))

  def test_lognormal_tails_are_in_order(self):
    dist = tfp.distributions.LogNormal(loc=4., scale=1.)
    self.assertGreater(
        helpers.upper_tail(dist, 2**-8), helpers.lower_tail(dist, 2**-8))

  def test_normal_tails_are_in_order(self):
    dist = tfp.distributions.Normal(loc=3., scale=5.)
    self.assertGreater(
        helpers.upper_tail(dist, 2**-8), helpers.lower_tail(dist, 2**-8))


if __name__ == "__main__":
  tf.test.main()
