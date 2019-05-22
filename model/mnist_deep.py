# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#Original File from: https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/examples/tutorials/mnist/mnist_deep.py
#Commit: 9f3bd2cf1eccdc76ed1934ade96c6cd4464bb8b2
#Available as of 22.05.19
#NOTICE: Changes have been made to the original File
#================================================================================
"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import numpy

FLAGS = None


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#No changes to the original up until here



from AgentFunction import AgentFunction
#Includes code from the original main function. Adaped to work with the AgentFunction 
class AgentMNISTFunction(AgentFunction):
    """
    Adaption of the mnist_deep model. 
    """
    def __init__(self, flags, data, labels, batch_size, gpu_nr, memory_fraction, testing, weigts = None, agent_nr=1):
        """
        :param flags: flags set by the user
        :param data: sampels if None have to be loaded in the by the actual implementation
        :param labels: lables  if None have to be loaded in the by the actual implementation
        :param batch_size: batch size to be used
        :param gpu_nr: Number of the GPU, the agent should run on
        :param memory_fraction: fraction of the GPU memory the agent should maximally use
        :param testing: True if we run a test set, False for training
        """
        super().__init__(flags, data, labels, batch_size, gpu_nr,memory_fraction, testing,weigts, agent_nr)

        self.data = data
        self.lables = labels
        self.batch_size = batch_size
        self.testing = testing

        conf = tf.ConfigProto()
        conf.use_per_session_threads = True
        conf.intra_op_parallelism_threads = 1
        conf.inter_op_parallelism_threads = 1  # 0 for default
        conf.allow_soft_placement = True
        conf.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        conf.gpu_options.allow_growth = True
        conf.gpu_options.visible_device_list = str(gpu_nr)

        self.graph = tf.Graph()
        self.session = tf.Session(config=conf, graph=self.graph)
        import time
        if data is None:
            with self.graph.as_default():
                # Import data
                seed = (int(time.time() * 100) * agent_nr) % (2 ** 32 - 1)
                tf.set_random_seed(seed)
                path = os.environ['HOME']
                path = path + '/mnist_data/'
                self.mnist = input_data.read_data_sets(path)
                
                # Create the model
                self.x = tf.placeholder(tf.float32, [None, 784])

                # Define loss and optimizer
                self.y_ = tf.placeholder(tf.int64, [None])

                # Build the graph for the deep net
                self.y_conv, self.keep_prob = deepnn(self.x)

                with tf.name_scope('loss'):
                    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                        labels=self.y_, logits=self.y_conv)
                self.cross_entropy = tf.reduce_mean(cross_entropy)

                with tf.name_scope('grad'):
                    self.grads = tf.gradients([cross_entropy], self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

                with tf.name_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), self.y_)
                    correct_prediction = tf.cast(correct_prediction, tf.float32)
                self.accuracy = tf.reduce_mean(correct_prediction)

                self.session.run(tf.global_variables_initializer())
        else:
            with self.graph.as_default():
                d_p = tf.placeholder(dtype=tf.float32, shape=data.shape)
                l_p = tf.placeholder(dtype=tf.int64, shape=labels.shape)
                if testing:
                    dataset = tf.data.Dataset.from_tensor_slices((d_p, l_p)).repeat().batch(self.batch_size)
                else:
                    dataset = tf.data.Dataset.from_tensor_slices((d_p, l_p)). \
                        shuffle(data.shape[0]).repeat().batch(self.batch_size)

                self.batch_iterator = dataset.make_initializable_iterator()

                batch = self.batch_iterator.get_next()
                self.x = batch[0]
                self.y_ = batch[1]

                # Build the graph for the deep net
                self.y_conv, self.keep_prob = deepnn(self.x)

                with tf.name_scope('loss'):
                    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                        labels=self.y_, logits=self.y_conv)
                self.cross_entropy = tf.reduce_mean(cross_entropy)

                with tf.name_scope('grad'):
                    self.grads = tf.gradients([cross_entropy],
                                              self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

                with tf.name_scope('accuracy'):
                    correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), self.y_)
                    correct_prediction = tf.cast(correct_prediction, tf.float32)
                self.accuracy = tf.reduce_mean(correct_prediction)
                self.session.run(tf.global_variables_initializer())
                self.session.run((self.batch_iterator.initializer), feed_dict={d_p: data, l_p: labels})

        self.trainable = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)



    def train(self, weights):
        """
        Do one more forward propagation and calculate the gardient
        :param new_weights: The weights to be loaded before beginning
        :return:    grads: calculated gradient, as numpy representation
                    globals: other global variables, as numpy representation
                    metrics: list of metrics
        """
        if self.data is None:
            with self.graph.as_default():
                for i, w in enumerate(self.trainable):
                    w.load(weights[i], session=self.session)

                batch =self.mnist.train.next_batch(self.batch_size)
                grads, loss, accuracy = self.session.run((self.grads, self.cross_entropy, self.accuracy),
                                                         feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

            return grads,  [loss,accuracy]
        else:
            with self.graph.as_default():
                for i, w in enumerate(self.trainable):
                    w.load(weights[i], session=self.session)

                grads, loss, accuracy = self.session.run((self.grads, self.cross_entropy, self.accuracy),
                                                         feed_dict={self.keep_prob: 0.5})

            return grads, [loss, accuracy]



    def get_weight(self):
        """
        :return: the current weight of the agent
        """
        with self.graph.as_default():
            t = self.session.run(self.trainable)
            return t

    def get_globals(self):
        """
        :return: the current global variables of the agent
        """
        return []

    def evaluate(self, weights, globals):
        """
        :param weights: weights to be loaded
        :param globals: global variables to be loaded
        :return: list of metrics
        """
        if self.data is None:
            for i, w in enumerate(self.trainable):
                w.load(weights[i], session=self.session)
            with self.graph.as_default():
                batch = self.mnist.test.next_batch(self.batch_size, shuffle=False)
                loss, accuracy = self.session.run((self.cross_entropy, self.accuracy), feed_dict={self.x: batch[0],
                                               self.y_: batch[1],
                                               self.keep_prob: 1.0})
                return [loss,accuracy]
        else:
            for i, w in enumerate(self.trainable):
                w.load(weights[i], session=self.session)
            with self.graph.as_default():
                loss, accuracy = self.session.run((self.cross_entropy, self.accuracy), feed_dict={self.keep_prob: 1.0})
                return [loss, accuracy]

    def close(self):
        self.session.close()
