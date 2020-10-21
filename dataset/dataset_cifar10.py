import tensorflow as tf
import numpy as np


def get_dataset_snr(snr_db):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_nums, test_nums = len(x_train), len(x_test)
    snrdb_train = snr_db * np.ones(shape=(train_nums,))
    snrdb_test = snr_db * np.ones(shape=(test_nums,))
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, snrdb_train), x_train))
    test_ds = tf.data.Dataset.from_tensor_slices(((x_test, snrdb_test), x_test))
    return (train_ds, train_nums), (test_ds, test_nums)


def get_dataset_snr_and_h(snr_db):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_nums, test_nums = len(x_train), len(x_test)
    snrdb_train = snr_db * np.ones(shape=(train_nums,))
    h_real_train = np.sqrt(1 / 2) * np.random.randn(train_nums, )
    h_imag_train = np.sqrt(1 / 2) * np.random.randn(train_nums, )
    snrdb_test = snr_db * np.ones(shape=(test_nums,))
    h_real_test = np.sqrt(1 / 2) * np.random.randn(test_nums, )
    h_imag_test = np.sqrt(1 / 2) * np.random.randn(test_nums, )
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, snrdb_train, h_real_train, h_imag_train), x_train))
    test_ds = tf.data.Dataset.from_tensor_slices(((x_test, snrdb_test, h_real_test, h_imag_test), x_test))
    return (train_ds, train_nums), (test_ds, test_nums)


def get_dataset_snr_range(snr_db_low, snr_db_high):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_nums, test_nums = len(x_train), len(x_test)
    snrdb_train = np.random.rand(train_nums, ) * (snr_db_high - snr_db_low) + snr_db_low
    snrdb_test = np.random.rand(test_nums, ) * (snr_db_high - snr_db_low) + snr_db_low
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, snrdb_train), x_train))
    test_ds = tf.data.Dataset.from_tensor_slices(((x_test, snrdb_test), x_test))
    return (train_ds, train_nums), (test_ds, test_nums)


def get_dataset_snr_range_and_h(snr_db_low, snr_db_high):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_nums, test_nums = len(x_train), len(x_test)
    h_real_train = np.sqrt(1 / 2) * np.random.randn(train_nums, )
    h_imag_train = np.sqrt(1 / 2) * np.random.randn(train_nums, )
    snrdb_train = np.random.rand(train_nums, ) * (snr_db_high - snr_db_low) + snr_db_low
    snrdb_test = np.random.rand(test_nums, ) * (snr_db_high - snr_db_low) + snr_db_low
    h_real_test = np.sqrt(1 / 2) * np.random.randn(test_nums, )
    h_imag_test = np.sqrt(1 / 2) * np.random.randn(test_nums, )
    train_ds = tf.data.Dataset.from_tensor_slices(((x_train, snrdb_train, h_real_train, h_imag_train), x_train))
    test_ds = tf.data.Dataset.from_tensor_slices(((x_test, snrdb_test, h_real_test, h_imag_test), x_test))
    return (train_ds, train_nums), (test_ds, test_nums)


def get_test_dataset_burst(snr_db, b_prob, b_stddev):
    cifar10 = tf.keras.datasets.cifar10
    (_, _), (x_test, _) = cifar10.load_data()
    test_nums = len(x_test)
    snrdb_test = snr_db * np.ones(shape=(test_nums,))
    b_prob_test = b_prob * np.ones(shape=(test_nums,))
    b_stddev_test = b_stddev * np.ones(shape=(test_nums,))
    test_ds = tf.data.Dataset.from_tensor_slices(((x_test, snrdb_test, b_prob_test, b_stddev_test), x_test))
    return test_ds, test_nums
