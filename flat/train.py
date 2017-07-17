from fileio import mnist_read
from flat import model

import tensorflow as tf
import numpy as np

# Todo: Save to file

N_EPOCHS = 10
BATCH_SIZE = 512

in_x = tf.placeholder(dtype=tf.float32, shape=[None, mnist_read.IMAGE_WIDTH * mnist_read.IMAGE_HEIGHT])
in_y = tf.placeholder(dtype=tf.float32, shape=[None, mnist_read.N_CLASSES])

out = model.create_flat_model(in_x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=in_y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

train_images, train_labels = mnist_read.parse_image_file('train'), mnist_read.parse_label_file('train')
num_examples = len(train_images)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for n in range(N_EPOCHS):
        for i in range(int(num_examples / BATCH_SIZE)):
            images = np.reshape(np.asarray(train_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]).flatten(),
                                (BATCH_SIZE, mnist_read.IMAGE_WIDTH * mnist_read.IMAGE_HEIGHT))
            labels = np.asarray(train_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            _, batch_loss = session.run([optimizer, loss], feed_dict={in_x: images, in_y: labels})
            # Todo: Print average loss to get a feel for progress
        print(n + 1, '/', N_EPOCHS, 'epochs completed')
