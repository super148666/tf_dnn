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
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.
    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.
    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.
    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


def cnn_layers(x_train_input):
    x = Conv2D(32, (3, 3),
                      activation='relu', padding='same')(x_train_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_train_out = Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return x_train_out


sess = K.get_session()


batch_size = 100
batch_shape = (batch_size, 32, 32, 1)
num_classes = 2
epochs = 5
capacity = 10000
min_after_dequeue = 3000
enqueue_many = True


data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_frontend_trained_model.h5'

train_data_path = './tfrecords/train.tfrecords'
train_num_examples = 0
for record in tf.python_io.tf_record_iterator(train_data_path):
    train_num_examples += 1
print(train_num_examples)
test_data_path = './tfrecords/test.tfrecords'
test_num_examples = 0
for record in tf.python_io.tf_record_iterator(test_data_path):
    test_num_examples += 1
print(test_num_examples)
valid_data_path = './tfrecords/valid.tfrecords'
valid_num_examples = 0
for record in tf.python_io.tf_record_iterator(valid_data_path):
    valid_num_examples += 1
print(valid_num_examples)

train_feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

test_feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

valid_feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

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



train_image = tf.decode_raw(train_features['image'], tf.float32)
train_image = tf.reshape(train_image, [32, 32])
train_label = tf.cast(train_features['label'], tf.int32)
print(train_image.shape,train_label.shape)
train_data, train_labels = tf.train.shuffle_batch([train_image, train_label], batch_size=batch_size, capacity=capacity, num_threads=8, min_after_dequeue=min_after_dequeue,enqueue_many=enqueue_many)
test_image = tf.decode_raw(test_features['image'], tf.float32)
test_image = tf.reshape(test_image, [32, 32])
test_label = tf.cast(test_features['label'], tf.int32)
test_data, test_labels = tf.train.shuffle_batch([test_image, test_label], batch_size=batch_size, capacity=capacity, num_threads=8, min_after_dequeue=min_after_dequeue,enqueue_many=enqueue_many)
valid_image = tf.decode_raw(valid_features['image'], tf.float32)
valid_image = tf.reshape(valid_image, [32, 32])
valid_label = tf.cast(valid_features['label'], tf.int32)
valid_data, valid_labels = tf.train.shuffle_batch([valid_image, valid_label], batch_size=batch_size, capacity=capacity, num_threads=8, min_after_dequeue=min_after_dequeue,enqueue_many=enqueue_many)

train_data = tf.cast(train_data, tf.float32)
train_data = tf.reshape(train_data, shape=batch_shape)
test_data = tf.cast(test_data, tf.float32)
test_data = tf.reshape(test_data, shape=batch_shape)
train_labels = tf.cast(train_labels, tf.int32)
train_labels = tf.one_hot(train_labels, num_classes)
test_labels = tf.cast(test_labels, tf.int32)
test_labels = tf.one_hot(test_labels, num_classes)

x_batch_shape = train_data.get_shape().as_list()
y_batch_shape = train_labels.get_shape().as_list()

model_input = layers.Input(tensor=train_data)
model_output = cnn_layers(model_input)
train_model = keras.models.Model(input=model_input, outputs=model_output)

train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[train_labels])
train_model.summary()

train_image.num_examples

x_test_batch_shape = test_data.get_shape().as_list()
y_test_batch_shape = test_labels.get_shape().as_list()

test_model_input = layers.Input(tensor=test_data)
test_model_output = cnn_layers(test_model_input)
test_model = keras.models.Model(inputs=test_model_input, outputs=test_model_output)

test_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'],
                    target_tensors=[test_labels])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

train_model.fit(epochs=epochs,
                steps_per_epoch=int(np.ceil(train_num_examples / float(batch_size))),
                callbacks=[EvaluateInputTensor(test_model, steps=100)])


train_model.save_weights(model_name)

coord.request_stop()
coord.join(threads)
K.clear_session()

x_test = np.reshape(test_data, (test_data.shape[0], 28, 28, 1))
y_test = test_labels
x_test_inp = layers.Input(shape=(x_test.shape[1:]))
test_out = cnn_layers(x_test_inp)
test_model = keras.models.Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights(model_name)
test_model.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(x_test,
                                keras.utils.to_categorical(y_test),
                                batch_size=batch_size)
print('\nTest accuracy: {0}'.format(acc))

# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess = tf.Session()
# print(sess.run(init_op))

# def cnn_model_fn(features, labels, mode):
#   """Model function for CNN."""
#   # Input Layer
#   # Reshape X to 4-D tensor: [batch_size, width, height, channels]
#   # MNIST images are 32x32 pixels, and have one color channel
#   input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])

#   # Convolutional Layer #1
#   # Computes 3 features using a 5x5 filter with ReLU activation.
#   # Padding is added to preserve width and height.
#   # Input Tensor Shape: [batch_size, 32, 32, 3]
#   # Output Tensor Shape: [batch_size, 32, 32, 3]
#   conv1 = tf.layers.conv2d(
#       inputs=input_layer,
#       filters=3,
#       kernel_size=[5, 5],
#       padding="same",
#       activation=tf.nn.relu)

#   # Pooling Layer #1
#   # First max pooling layer with a 2x2 filter and stride of 2
#   # Input Tensor Shape: [batch_size, 32, 32, 3]
#   # Output Tensor Shape: [batch_size, 16, 16, 3]
#   pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

#   # Convolutional Layer #2
#   # Computes 6 features using a 5x5 filter.
#   # Padding is added to preserve width and height.
#   # Input Tensor Shape: [batch_size, 16, 16, 3]
#   # Output Tensor Shape: [batch_size, 16, 16, 6]
#   conv2 = tf.layers.conv2d(
#       inputs=pool1,
#       filters=6,
#       kernel_size=[5, 5],
#       padding="same",
#       activation=tf.nn.relu)

#   # Pooling Layer #2
#   # Second max pooling layer with a 2x2 filter and stride of 2
#   # Input Tensor Shape: [batch_size, 16, 16, 6]
#   # Output Tensor Shape: [batch_size, 8, 8, 6]
#   pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#   # Flatten tensor into a batch of vectors
#   # Input Tensor Shape: [batch_size, 8, 8, 6]
#   # Output Tensor Shape: [batch_size, 8 * 8 * 6]
#   pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 6])

#   # Dense Layer
#   # Densely connected layer with 120 neurons
#   # Input Tensor Shape: [batch_size, 8 * 8 * 6]
#   # Output Tensor Shape: [batch_size, 120]
#   dense = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)

#   # Add dropout operation; 0.6 probability that element will be kept
#   dropout = tf.layers.dropout(
#       inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

#   # Logits layer
#   # Input Tensor Shape: [batch_size, 120]
#   # Output Tensor Shape: [batch_size, 10]
#   logits = tf.layers.dense(inputs=dropout, units=1)

#   predictions = {
#       # Generate predictions (for PREDICT and EVAL mode)
#       "classes": tf.argmax(input=logits, axis=1),
#       # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#       # `logging_hook`.
#       "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#   }
#   if mode == tf.estimator.ModeKeys.PREDICT:
#     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

#   # Calculate Loss (for both TRAIN and EVAL modes)
#   loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

#   # Configure the Training Op (for TRAIN mode)
#   if mode == tf.estimator.ModeKeys.TRAIN:
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#     train_op = optimizer.minimize(
#         loss=loss,
#         global_step=tf.train.get_global_step())
#     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

#   # Add evaluation metrics (for EVAL mode)
#   eval_metric_ops = {
#       "accuracy": tf.metrics.accuracy(
#           labels=labels, predictions=predictions["classes"])}
#   return tf.estimator.EstimatorSpec(
#       mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# def main(unsaved_argv):
#     classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

#     tensors_to_log = {"probabilities": "softmax_tensor"}
#     logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
#     # print(train_data.shape)
#     # print(train_data.format)
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x":train_data},
#         y=train_labels,
#         batch_size=batch_size,
#         num_epochs=epochs,
#         shuffle=True)

#     classifier.train(
#         input_fn=train_input_fn,
#         steps=20000,
#         hooks=[logging_hook])

#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": test_data},
#         y=test_labels,
#         num_epochs=1,
#         shuffle=False)
#     eval_results = classifier.evaluate(input_fn=eval_input_fn)
#     print(eval_results)


# if __name__ == "__main__":
#     tf.app.run()