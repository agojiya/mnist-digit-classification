import os

from fileio import mnist_read
from flat import model

import tensorflow as tf
import numpy as np

# Todo: Read from file
SAVE_PATH = 'X:/mnist/model/flat/model-epoch'
SAVE_DIR = '/'.join(SAVE_PATH.split('/')[0:-1])
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

N_EPOCHS = 20
BATCH_SIZE = 512

in_x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_read.IMAGE_WIDTH * mnist_read.IMAGE_HEIGHT])
in_y = tf.placeholder(dtype=tf.float32, shape=[None, mnist_read.N_CLASSES])

out = model.create_flat_model(in_x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=in_y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

train_images, train_labels = mnist_read.parse_image_file('train'), mnist_read.parse_label_file('train')
num_examples = len(train_images)

saver = tf.train.Saver(max_to_keep=None)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for e in range(N_EPOCHS):
        for i in range(int(num_examples / BATCH_SIZE) + 1):
            n = min(BATCH_SIZE, num_examples - BATCH_SIZE * i)
            images = np.reshape(np.asarray(train_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]).flatten(),
                                (n, mnist_read.IMAGE_WIDTH * mnist_read.IMAGE_HEIGHT))
            labels = np.asarray(train_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            _, batch_loss = session.run([optimizer, loss], feed_dict={in_x: images, in_y: labels})
            # Todo: Print average loss to get a feel for progress
        if (e + 1) % 5 == 0:
            saver.save(sess=session, save_path=SAVE_PATH, global_step=(e + 1))
        print(e + 1, '/', N_EPOCHS, 'epochs completed')
