import os, sys

from fileio import mnist_read, misc
from conv2d import model

import tensorflow as tf
import numpy as np

BATCH_SIZE = 512

SAVE_PATH = 'X:/mnist/model/conv2d/model-epoch'
SAVE_DIR = '/'.join(SAVE_PATH.split('/')[0:-1])
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

in_x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_read.IMAGE_WIDTH, mnist_read.IMAGE_HEIGHT])

out = model.create_conv2d_model(in_x)

test_images, test_labels = mnist_read.parse_image_file('test'), mnist_read.parse_label_file('test')
num_examples = len(test_images)

saver = tf.train.Saver()
saved_epochs = misc.get_highest_epoch_saved(SAVE_DIR)
if saved_epochs == 0:
    sys.exit('Could not find save file')
with tf.Session() as session:
    saver.restore(sess=session, save_path=(SAVE_PATH + '-' + str(saved_epochs)))
    print('Loaded', saved_epochs, 'epochs of training')

    correct_images, incorrect_images = [], []
    for i in range(int(num_examples / BATCH_SIZE) + 1):
        n = min(BATCH_SIZE, num_examples - BATCH_SIZE * i)
        images = np.array(test_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        actual_labels = np.array(test_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        predicted_labels = session.run(out, feed_dict={in_x: images})

        for x in range(len(actual_labels)):
            if np.argmax(actual_labels[x]) == np.argmax(predicted_labels[x]):
                correct_images.append(images[x])
            else:
                incorrect_images.append(images[x])

    print('Accuracy:', '{}%'.format(round(len(correct_images) / num_examples * 100, ndigits=2)))
