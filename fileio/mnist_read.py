import numpy as np

BASE_PATH = 'X:/mnist/{}/{}'
IMAGE_WIDTH = IMAGE_HEIGHT = 28
N_CLASSES = 10


def parse_label_file(data_type):
    path = BASE_PATH.format(data_type, 'labels.idx1-ubyte')
    with open(file=path, mode='rb') as file:
        magic_number = int.from_bytes(file.read(4), byteorder='big')
        if magic_number != 2049:
            import sys
            sys.exit('Invalid file (magic number mismatch)')
        else:
            n_labels = int.from_bytes(file.read(4), byteorder='big')
            buffer = file.read(n_labels)
            labels = np.frombuffer(buffer=buffer, dtype=np.uint8)
    return np.eye(N=10)[labels]


def parse_image_file(data_type):
    path = BASE_PATH.format(data_type, 'images.idx3-ubyte')
    with open(file=path, mode='rb') as file:
        magic_number = int.from_bytes(file.read(4), byteorder='big')
        if magic_number != 2051:
            import sys
            sys.exit('Invalid file (magic number mismatch)')
        else:
            import os
            n_images = int.from_bytes(file.read(4), byteorder='big')
            file.seek(8, os.SEEK_CUR)  # Skip 8 bytes (image width and height already defined)
            buffer = file.read(n_images * IMAGE_WIDTH * IMAGE_HEIGHT)
            pixels = np.frombuffer(buffer=buffer, dtype=np.uint8)
    out = pixels.reshape((n_images, IMAGE_WIDTH, IMAGE_HEIGHT))
    # In the data set 0=white and 255=black so we have to invert it
    out = np.abs(out.astype(dtype=np.int8) - 255).astype(dtype=np.uint8)
    return out

if __name__ == '__main__':
    images, labels = parse_image_file('train'), parse_label_file('train')

    import random
    index = random.randrange(0, len(images))

    import cv2
    cv2.imshow('Training image', images[index])
    print(np.argmax(labels[index]))
    cv2.waitKey()
