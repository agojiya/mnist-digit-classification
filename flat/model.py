

def create_flat_model(data_in):
    """Creates the dense, connected layers and returns the output layer.

    Args:
        data_in (Tensor): A tensor of shape [-1, 784] representing the flattened image (28 * 28 = 784)

    Returns:
        Tensor: A tensor of shape [-1, 10] representing the output of the neural network (10 possible classes)

    Todo:
        * Create the dense layers and return the final dense layer
    """
