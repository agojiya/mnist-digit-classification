from os import path, makedirs
from random import shuffle

from fileio import mnist_read
from conv2d import model

import tensorflow as tf

SAVE_PATH = 'X:/mnist/model/flat/model-epoch'
SAVE_DIR = '/'.join(SAVE_PATH.split('/')[0:-1])
if not path.isdir(SAVE_DIR):
    makedirs(SAVE_DIR)

N_EPOCHS = 10
BATCH_SIZE = 512

in_x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_read.IMAGE_WIDTH, mnist_read.IMAGE_HEIGHT])
in_y = tf.placeholder(dtype=tf.float32, shape=[None, mnist_read.N_CLASSES])

out = model.create_conv2d_model(in_x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=in_y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

train_images, train_labels = mnist_read.parse_image_file('train'), mnist_read.parse_label_file('train')
num_examples = len(train_images)
zipped = list(zip(train_images, train_labels))
shuffle(zipped)
train_images, train_labels = zip(*zipped)

# Todo: Saver and training
