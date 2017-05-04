import matplotlib
matplotlib.use('Agg')
# %% imports
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['image.cmap'] = 'gray'
import tensorflow as tf
import os
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import math
import time
import scipy
import shutil
import scipy.misc
from tensorflow.contrib import slim

model_path = "/home/vkslm/ckpt/model.ckpt"

h = 150
w = 383


def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.

    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def load_dataset(batch_size, num_epochs):
    def read_labeled_image_list():
        """Reads a .txt file containing pathes and labeles
        Args:
           image_list_file: a .txt file with one /path/to/image per line
           label: optionally, if set label will be pasted after each line
        Returns:
           List with all filenames in file image_list_file
        """
        off_fns = []
        on_fns = []
        for i in range(23003):
            off_fns.append('%010d-off.png' % i)
            on_fns.append('%010d-on.png' % i)

        label_fns = [os.path.join('out/', f) for f in off_fns]
        data_fns = [os.path.join('out/', f) for f in on_fns]

        return data_fns, label_fns

    def read_images_from_disk(input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """

        data_content = tf.read_file(input_queue[0])
        label_content = tf.read_file(input_queue[1])

        data = tf.image.decode_png(data_content, channels=1)
        label = tf.image.decode_png(label_content, channels=1)

        data.set_shape([h, w, 1])
        data = tf.cast(data, tf.float32) / 255.0

        label.set_shape([h, w, 1])
        label = tf.cast(label, tf.float32) / 255.0

        return data, label

    # Reads pfathes of images together with their labels
    image_list, label_list = read_labeled_image_list()

    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.string)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    image, label = read_images_from_disk(input_queue)

    # Optional Image and Label Batching
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size, num_threads=4,
                                              capacity=4 * batch_size,
                                              name='input')

    return image_batch, label_batch

input_shape = [None, h * w]

x = tf.placeholder(
    tf.float32, input_shape, name='x')
y_true = tf.placeholder(
    tf.float32, input_shape, name='y')
x_tensor = tf.reshape(x, [-1, h, w, 1])
y_true_tensor = tf.reshape(y_true, [-1, h, w, 1])

prob = 0.60
with slim.arg_scope([slim.conv2d], padding='SAME', normalizer_fn=slim.batch_norm,
                    normalizer_params={'decay': 0.9997, 'is_training': batch_norm, 'updates_collections': None,
                                       'trainable': is_training},
                    weights_initializer=slim.initializers.xavier_initializer(), weights_regularizer=slim.l2_regularizer(0.0005)):

conv1 = slim.repeat(x_tensor, 2, slim.conv2d, 32, [3, 3], scope='conv1')
conv1 = slim.dropout(conv1, prob, is_training=is_training, scope='dropout1')
pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')  # 64
conv2 = slim.repeat(pool1, 2, slim.conv2d, 64, [3, 3], scope='conv2')
conv2 = slim.dropout(conv2, prob, is_training=is_training, scope='dropout2')
pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')  # 32
conv3 = slim.repeat(pool2, 2, slim.conv2d, 128, [3, 3], scope='conv3')
conv3 = slim.dropout(conv3, prob, is_training=is_training, scope='dropout3')
pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')  # 16
conv4 = slim.repeat(pool3, 2, slim.conv2d, 256, [3, 3], scope='conv4')
conv4 = slim.dropout(conv4, prob, is_training=is_training, scope='dropout4')
pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')  # 8
conv5 = slim.repeat(pool4, 2, slim.conv2d, 512, [3, 3], scope='conv5')
conv5 = slim.dropout(conv5, prob, is_training=is_training, scope='dropout5')

with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'decay': 0.9997, 'is_training': batch_norm, 'updates_collections': None,
                                       'trainable': is_training},
                    weights_initializer=slim.initializers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(0.0005)):

deconv1 = slim.conv2d_transpose(conv5, 512, stride=2, kernel_size=2)  # 16
deconv1 = slim.dropout(
    deconv1, prob, is_training=is_training, scope='d_dropout1')
concat1 = tf.concat(3, [conv4, deconv1], name='concat1')
conv6 = slim.repeat(concat1, 2, slim.conv2d, 256, [3, 3], scope='conv6')
conv6 = slim.dropout(conv6, prob, is_training=is_training, scope='dropout6')
deconv2 = slim.conv2d_transpose(conv6, 256, stride=2, kernel_size=2)  # 32
deconv2 = slim.dropout(
    deconv2, prob, is_training=is_training, scope='d_dropout2')
concat2 = tf.concat(3, [conv3, deconv2], name='concat2')
conv7 = slim.repeat(concat2, 2, slim.conv2d, 128, [3, 3], scope='conv7')
conv7 = slim.dropout(conv7, prob, is_training=is_training, scope='dropout7')
deconv3 = slim.conv2d_transpose(conv7, 128, stride=2, kernel_size=2)  # 64
deconv3 = slim.dropout(
    deconv3, prob, is_training=is_training, scope='d_dropout3')
concat3 = tf.concat(3, [conv2, deconv3], name='concat3')
conv8 = slim.repeat(concat3, 2, slim.conv2d, 64, [3, 3], scope='conv8')
conv8 = slim.dropout(conv8, prob, is_training=is_training, scope='dropout8')
deconv4 = slim.conv2d_transpose(conv8, 64, stride=2, kernel_size=2)  # 128
deconv4 = slim.dropout(
    deconv4, prob, is_training=is_training, scope='d_dropout4')
concat4 = tf.concat(3, [conv1, deconv4], name='concat4')
conv9 = slim.repeat(concat4, 2, slim.conv2d, 32, [3, 3], scope='conv9')
conv9 = slim.dropout(conv9, prob, is_training=is_training, scope='dropout9')
current_input = slim.conv2d(
    conv9, 2, [1, 1], activation_fn=tf.nn.sigmoid, scope='conv1x1')


# %%
# now have the reconstruction through the network
y = current_input
# cost function measures pixel-wise difference
cost = tf.reduce_sum(tf.square(y - y_true_tensor))
# %%
learning_rate = 0.001

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# %%
# We create a session to use the graph
sess = tf.Session()

# %%
# Fit all training data
batch_size = 8
n_epochs = 100

batch_data, batch_label = load_dataset(batch_size, n_epochs)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
tf.train.start_queue_runners(sess=sess)
saver = tf.train.Saver()

exp_folder = 'results_' + time.strftime('%d_%b_%H-%M-%S')
os.mkdir(exp_folder)
checkpoint_path = os.path.join(exp_folder, 'model.ckpt')
for epoch_i in range(n_epochs):
    for batch_i in range(20000 // batch_size):
        batch_data_flatten = tf.reshape(batch_data, [batch_size, h * w])
        batch_label_flatten = tf.reshape(batch_label, [batch_size, h * w])

        batch_xs, batch_ys = sess.run(
            [batch_data_flatten, batch_label_flatten])
        sess.run(optimizer, feed_dict={x: batch_xs, y_true: batch_ys})
        print((batch_i / (20000 // batch_size)))
    print(epoch_i, sess.run(cost, feed_dict={x: batch_xs, y_true: batch_ys}))
    saver.save(sess, checkpoint_path, global_step=epoch_i)

    folder = exp_folder + '/%05d' % epoch_i

    os.mkdir(folder)
    # shutil.copy2('/home/vkslm/playground/index-vae.html', folder+'/index.html')
    recon = sess.run(y, feed_dict={x: batch_xs, y_true: batch_ys})
    for i in range(recon.shape[0]):
        plt.subplot(311)
        plt.imshow(recon[i].reshape((h, w)))
        plt.subplot(312)
        plt.imshow(batch_ys[i].reshape(h, w))
        plt.subplot(313)
        plt.imshow(batch_xs[i].reshape(h, w))

        filename = '%d.png' % (i)

        filename = os.path.join(folder, filename)
        plt.savefig(filename)
