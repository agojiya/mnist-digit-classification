import tensorflow as tf
from fileio import mnist_read

CONSTANT_255 = tf.constant(255, dtype=tf.float32)


def create_conv2d_model(data_in):
    """Creates the model consisting of convolutional layers, fully connected layers, and the output layer (returned).

    Args:
        data_in (Tensor): A Tensor of shape [-1, 28, 28] representing the image (in a batch)

    Returns:
        Tensor: A Tensor of shape [-1, 10] representing the output of the neural network (10 possible classes)

    Todo:
        Tweak as necessary
    """
    data_in = tf.reshape(data_in, [-1, mnist_read.IMAGE_WIDTH, mnist_read.IMAGE_HEIGHT, 1])

    data_in_padded = tf.add(
        tf.pad(tf.subtract(data_in, CONSTANT_255), [[0, 0], [2, 2], [2, 2], [0, 0]]), CONSTANT_255)
    c1 = tf.layers.conv2d(inputs=data_in_padded, filters=32, kernel_size=4, padding="valid", activation=tf.nn.relu)
    p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2, padding="valid", name='Pool-1')

    c2 = tf.layers.conv2d(inputs=p1, filters=32, kernel_size=4, padding="same", activation=tf.nn.relu)
    p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2, padding="same", name='Pool-2')

    c3 = tf.layers.conv2d(inputs=p2, filters=32, kernel_size=4, padding="same", activation=tf.nn.relu)
    p3 = tf.layers.max_pooling2d(inputs=c3, pool_size=2, strides=2, padding="same", name='Pool-3')

    c4 = tf.layers.conv2d(inputs=p3, filters=32, kernel_size=4, padding="same", activation=tf.nn.relu)
    p4 = tf.layers.max_pooling2d(inputs=c4, pool_size=2, strides=2, padding="same", name='Pool-4')

    # >>> p4.shape
    # (-1, 2, 2, 32)
    # 28 / 2 = 14 / 2 = 7 / 2 = 4 / 2 = 2 (Result of 4 max_pooling2d layers with pool_size=2 and strides=2)
    # (Rounded up since we used padding="same")
    reshaped_p4 = tf.reshape(p4, [-1, 2 * 2 * 32])
    d1 = tf.layers.dense(inputs=reshaped_p4, units=512, activation=tf.nn.relu, use_bias=True)
    d2 = tf.layers.dense(inputs=d1, units=512, activation=tf.nn.relu, use_bias=True)
    d2_normalized = tf.divide(d2, tf.reduce_max(tf.abs(d2)))
    out = tf.layers.dense(inputs=d2_normalized, units=mnist_read.N_CLASSES, activation=tf.nn.sigmoid)
    return out
