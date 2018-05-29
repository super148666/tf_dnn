from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

import keras
from keras import backend as K
from keras.callbacks import Callback
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os



batch_size = 2
num_classes = 2
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_frontend_trained_model.h5'

train_data_path = './tfrecords/train.tfrecords'
test_data_path = './tfrecords/test.tfrecords'
valid_data_path = './tfrecords/valid.tfrecords'

train_feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}

test_feature = {'test/image': tf.FixedLenFeature([], tf.string),
               'test/label': tf.FixedLenFeature([], tf.int64)}

valid_feature = {'valid/image': tf.FixedLenFeature([], tf.string),
               'valid/label': tf.FixedLenFeature([], tf.int64)}

train_filename_queue = tf.train.string_input_producer([train_data_path], num_epochs=1)
test_filename_queue = tf.train.string_input_producer([test_data_path], num_epochs=1)
valid_filename_queue = tf.train.string_input_producer([valid_data_path], num_epochs=1)

reader = tf.TFRecordReader()
_, train_example = reader.read(train_filename_queue)
_, test_example = reader.read(test_filename_queue)
_, valid_example = reader.read(valid_filename_queue)

train_features = tf.parse_single_example(train_example, features=train_feature)
test_features = tf.parse_single_example(test_example, features=test_feature)
valid_features = tf.parse_single_example(valid_example, features=valid_feature)

train_image = tf.decode_raw(train_features['train/image'], tf.float32)
print(train_image)
train_image = tf.reshape(train_image, [32, 32])
train_label = tf.cast(train_features['train/label'], tf.int32)
train_data, train_labels = tf.train.shuffle_batch([train_image, train_label], batch_size=batch_size, capacity=50000, num_threads=4, min_after_dequeue=10000)
test_image = tf.decode_raw(test_features['test/image'], tf.float32)
test_image = tf.reshape(test_image, [32, 32])
test_label = tf.cast(test_features['test/label'], tf.int32)
test_data, test_labels = tf.train.shuffle_batch([test_image, test_label], batch_size=batch_size, capacity=50000, num_threads=4, min_after_dequeue=10000)
valid_image = tf.decode_raw(valid_features['valid/image'], tf.float32)
valid_image = tf.reshape(valid_image, [32, 32])
valid_label = tf.cast(valid_features['valid/label'], tf.int32)
valid_data, valid_labels = tf.train.shuffle_batch([valid_image, valid_label], batch_size=batch_size, capacity=50000, num_threads=4, min_after_dequeue=10000)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
print(sess.run(init_op))
# print(sess.run([train_data,train_labels]))

# print('x_train shape:', train_data.shape)
# print(train_data.shape[0], 'train samples')
# print(test_data.shape[0], 'test samples')








def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 32x32 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])

  # Convolutional Layer #1
  # Computes 3 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 3]
  # Output Tensor Shape: [batch_size, 32, 32, 3]
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


def main(unsaved_argv):
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # print(train_data.shape)
    # print(train_data.format)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)

    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()