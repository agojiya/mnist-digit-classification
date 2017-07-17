import tensorflow as tf
from fileio import mnist_read


def create_flat_model(data_in):
    """Creates the dense, connected layers and returns the output layer.

    Args:
        data_in (Tensor): A Tensor of shape [-1, 784] representing the flattened image (28 * 28 = 784)

    Returns:
        Tensor: A Tensor of shape [-1, 10] representing the output of the neural network (10 possible classes)
    """
    d1 = tf.layers.dense(inputs=data_in, units=512, activation=tf.nn.relu, use_bias=True)
    d2 = tf.layers.dense(inputs=d1, units=256, activation=tf.nn.relu, use_bias=True)
    d3 = tf.layers.dense(inputs=d2, units=128, activation=tf.nn.relu, use_bias=True)
    d4 = tf.layers.dense(inputs=d3, units=64, activation=tf.nn.relu, use_bias=True)
    d5 = tf.layers.dense(inputs=d4, units=32, activation=tf.nn.relu, use_bias=True)
    d6 = tf.layers.dense(inputs=d5, units=16, activation=tf.nn.relu, use_bias=True)
    out = tf.layers.dense(inputs=d6, units=mnist_read.N_CLASSES)
    return out
