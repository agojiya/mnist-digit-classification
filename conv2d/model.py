

def create_conv2d_model(data_in):
    """Creates the model consisting of convolutional layers, fully connected layers, and the output layer (returned).

    Args:
        data_in (Tensor): A Tensor of shape [-1, 28, 28] representing the image (in a batch)

    Returns:
        Tensor: A Tensor of shape [-1, 10] representing the output of the neural network (10 possible classes)
    """
