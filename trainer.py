from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import os.path
import re
import sys
import tarfile
from six.moves import urllib

FLAGS = None

data_path = './tfrecords/train.tfrecords'  # address to save the hdf5 file
feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}

tf.logging.set_verbosity(tf.logging.INFO)

path_cones = '/media/chao/RAID1_L/chaoz/cone_detection_data/basler32/1/'
num_cones = 
path_non_cones = '/media/chao/RAID1_L/chaoz/cone_detection_data/basler32/0/'

from PIL import Image
from .utils import dense_to_one_hot

def _read_pngs_from(path):
  """Reads directory of images.
  Args:
    path: path to the directory
  Returns:
    A list of all images in the directory in the TF format (You need to call sess.run() or .eval() to get the value).
  """
  images = []
  png_files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
  for filename in png_files_path:
    im = Image.open(filename)
    im = np.asarray(im, np.uint8)
1
    # get only images name, not path
    image_name = filename.split('/')[-1].split('.')[0]
    images.append([int(image_name), im])

  images = sorted(images, key=lambda image: image[0])

  images_only = [np.asarray(image[1], np.uint8) for image in images]  # Use unint8 or you will be !!!
  images_only = np.array(images_only)

  print(images_only.shape)
  return images_only


def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [32, 32])
  return image_resized, label


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 32x32 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])

  # Convolutional Layer #1
  # Computes 3 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 1]
  # Output Tensor Shape: [batch_size, 32, 32, 1]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=3,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32, 32, 3]
  # Output Tensor Shape: [batch_size, 16, 16, 3]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 6 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 16, 3]
  # Output Tensor Shape: [batch_size, 16, 16, 6]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=6,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 16, 16, 6]
  # Output Tensor Shape: [batch_size, 8, 8, 6]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 8, 8, 6]
  # Output Tensor Shape: [batch_size, 8 * 8 * 6]
  pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 6])

  # Dense Layer
  # Densely connected layer with 120 neurons
  # Input Tensor Shape: [batch_size, 8 * 8 * 6]
  # Output Tensor Shape: [batch_size, 120]
  dense = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 120]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=1)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features=feature)
  image = tf.decode_raw(features['train/image'], tf.float32)
  label = tf.cast(features['train/label'], tf.int32)
  image = tf.reshape(image, [32, 32])
  train_data, train_labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
#   mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#   train_data = mnist.train.images  # Returns np.array
#   train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()