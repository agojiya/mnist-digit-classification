import tensorflow as tf


def create_conv2d_model(data_in):
    """Creates the model consisting of convolutional layers, fully connected layers, and the output layer (returned).

    Args:
        data_in (Tensor): A Tensor of shape [-1, 28, 28] representing the image (in a batch)

    Returns:
        Tensor: A Tensor of shape [-1, 10] representing the output of the neural network (10 possible classes)
    """
    c1 = tf.layers.conv2d(inputs=data_in, filters=32, kernel_size=4, padding="same", activation=tf.nn.relu)
    p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=2, strides=2, padding="same")

    c2 = tf.layers.conv2d(inputs=p1, filters=32, kernel_size=4, padding="same", activation=tf.nn.relu)
    p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=2, strides=2, padding="same")

    c3 = tf.layers.conv2d(inputs=p2, filters=32, kernel_size=4, padding="same", activation=tf.nn.relu)
    p3 = tf.layers.max_pooling2d(inputs=c3, pool_size=2, strides=2, padding="same")

    c4 = tf.layers.conv2d(inputs=p3, filters=32, kernel_size=4, padding="same", activation=tf.nn.relu)
    p4 = tf.layers.max_pooling2d(inputs=c4, pool_size=2, strides=2, padding="same")

    # Todo: Fully connected (dense) layers and output layer
