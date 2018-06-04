from __future__ import division, print_function, absolute_import

import tensorflow as tf

import os
import random

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

seed = 100

image_path = [
            #   "/media/chao/RAID1_L/chaoz/cone_detection_data/basler32/",
              "/media/chao/RAID1_L/chaoz/cone_detection_data/basler_new/",
            #   "/media/chao/RAID1_L/chaoz/cone_detection_data/basler/",
            #   "/media/chao/RAID1_L/chaoz/cone_detection_data/webcam/"
            ]

summary_path = '/home/chao/vision_ws/src/tensorflow_dnn/summary/'

model_path = '/home/chao/vision_ws/src/tensorflow_dnn/my_tf_model'


# Image Parameters
image_width = 32
image_height = 32
image_channel = 3

# Dataset Partition
train_partition = 0.4
valid_partition = 0.4
test_partition = 1.0 - valid_partition - train_partition

# Training Parameters
learning_rate = 0.001
num_steps = 2500
batch_size = 128
display_step = 10


# Network Parameters
num_input = image_width * image_height * image_channel  # MNIST data input (img shape: 28*28)
num_classes = 2  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units
output_size_1 = 6
output_size_2 = 12
output_size_3 = 120
output_size_4 = num_classes

# # tf Graph input
# X = tf.placeholder(tf.float32, [None, num_input])
# Y = tf.placeholder(tf.float32, [None, num_classes])
# keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_paths, batch_size):
    imagepaths, labels = list(), list()
    # An ID will be affected to each sub-folders by alphabetical order
    total_data = 0
    # List the directory
    for dataset_path in dataset_paths:
        label = 0
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # except StopIteration:
        #     pass
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # except StopIteration:
            #     pass
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps png images
                if sample.endswith('.png'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1

    random.Random(seed).shuffle(labels)
    random.Random(seed).shuffle(imagepaths)
    num_classes = label
    total_data = len(labels)

    train_size = (int)(total_data * train_partition) - 1
    valid_size = train_size + (int)(total_data * valid_partition) - 1
    test_size = valid_size + (int)(total_data * test_partition) - 1

    train_imagepaths = imagepaths[0:train_size]
    train_labels = labels[0:train_size]

    valid_imagepaths = imagepaths[train_size + 1:valid_size]
    valid_labels = labels[train_size + 1:valid_size]

    test_imagepaths = imagepaths[valid_size + 1:test_size]
    test_labels = labels[valid_size + 1:test_size]

    train_size = len(train_labels)
    valid_size = len(valid_labels)
    test_size = len(test_labels)
    print("train set size: ", train_size)
    print("valid set size: ", valid_size)
    print("test set size: ", test_size)

    # Convert to Tensor
    train_imagepaths = tf.convert_to_tensor(train_imagepaths, dtype=tf.string)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)

    valid_imagepaths = tf.convert_to_tensor(valid_imagepaths, dtype=tf.string)
    valid_labels = tf.convert_to_tensor(valid_labels, dtype=tf.int32)

    test_imagepaths = tf.convert_to_tensor(test_imagepaths, dtype=tf.string)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)

    # Build a TF Queue, shuffle data
    train_image, train_label = tf.train.slice_input_producer([train_imagepaths, train_labels],
                                                             shuffle=True)

    valid_image, valid_label = tf.train.slice_input_producer([valid_imagepaths, valid_labels],
                                                             shuffle=True)

    test_image, test_label = tf.train.slice_input_producer([test_imagepaths, test_labels],
                                                           shuffle=True)

    # Read images from disk
    train_image = tf.read_file(train_image)
    train_image = tf.image.decode_png(train_image, channels=image_channel)

    valid_image = tf.read_file(valid_image)
    valid_image = tf.image.decode_png(valid_image, channels=image_channel)

    test_image = tf.read_file(test_image)
    test_image = tf.image.decode_png(test_image, channels=image_channel)

    # Resize images to a common size
    train_image = tf.image.resize_images(train_image, [image_height, image_width])

    valid_image = tf.image.resize_images(valid_image, [image_height, image_width])

    test_image = tf.image.resize_images(test_image, [image_height, image_width])

    # Normalize
    train_image = train_image * 2.0 - 1.0

    valid_image = valid_image * 2.0 - 1.0

    test_image = test_image * 2.0 - 1.0

    # Create batches
    X, Y = tf.train.batch([train_image, train_label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    Xv, Yv = tf.train.batch([valid_image, valid_label], batch_size=valid_size,
                            capacity=batch_size * 8,
                            num_threads=4)

    Xt, Yt = tf.train.batch([test_image, test_label], batch_size=test_size,
                            capacity=batch_size * 8,
                            num_threads=4)

    return X, Y, Xv, Yv, Xt, Yt


# def split_dataset(images, labels, train_ratio, valid_ratio, test_ratio, shuffle=True)


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# Create some wrappers for simplicity
def conv2d(x, W, b, layer_name, strides=1, act=tf.nn.relu):
    # Conv2D wrapper, with bias and relu activation
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            variable_summaries(W)
            # W_max = tf.reduce_max(W)
            # W_min = tf.reduce_min(W)
            # W_0_to_1 = (W - W_min) / (W_max - W_min)
            # W_transposed = tf.transpose(W_0_to_1, [-1, 5, 5, 1])
            # tf.summary.image('filters', W_transposed)
        with tf.name_scope('biases'):
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):    
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            tf.summary.histogram('pre_activations',x)
        activations = act(x, name='activation')
        tf.summary.histogram('activations', activations)
    return activations


def maxpool2d(x, layer_name, k=2):
    # MaxPool2D wrapper
    with tf.name_scope(layer_name):
        out = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME', name='max_pooling')
        tf.summary.histogram('max_poolings', out)
    return out


def fc1d(x, W, b, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            variable_summaries(W)
            # W_max = tf.reduce_max(W)
            # W_min = tf.reduce_min(W)
            # W_0_to_1 = (W - W_min) / (W_max - W_min)
            # W_transposed = tf.transpose(W_0_to_1, [-1, 5, 5, output_size_1])
            # tf.image_summary('filters', W_transposed)
        with tf.name_scope('biases'):
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):    
            x_reshaped = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
            pre_activations = tf.add(tf.matmul(x_reshaped, W), b)
            tf.summary.histogram('pre_activations',pre_activations)
        activations = act(pre_activations, name='activation')
        tf.summary.histogram('activations', activations)

    return activations


def dropout1d(x, dropout, layer_name):
    with tf.name_scope(layer_name):
        tf.summary.scalar('dropout', dropout)
        dropped = tf.nn.dropout(x, dropout)

    return dropped


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, image_height, image_width, image_channel])

    # Convolution Layer
    # input size: 32x32x3
    # output size: 32x32x3
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], layer_name='conv1')
    # Max Pooling (down-sampling)
    # input size: 32x32x3
    # output size: 16x16x3
    conv1 = maxpool2d(conv1, k=2, layer_name='max_pool1')

    # Convolution Layer
    # input size: 16x16x3
    # output size: 16x16x6
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], layer_name='conv2')
    # Max Pooling (down-sampling)
    # input size: 16x16x6
    # output size: 8x8x6
    conv2 = maxpool2d(conv2, k=2, layer_name='max_pool2')

    # Fully connected layer
    # input size: 8x8x6
    # output size: 1024
    # Reshape conv2 output to fit fully connected layer input
    fc1 = fc1d(conv2, weights['wd1'], biases['bd1'], layer_name='fc1')
    
    # Dropout layer
    dp1 = dropout1d(fc1, dropout, 'dp1')

    # Output, class prediction
    # input size: 1024
    # output size: 2
    out = fc1d(dp1, weights['out'], biases['out'], layer_name='out', act=tf.identity)
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 6 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, image_channel, output_size_1])),
    # 5x5 conv, 6 inputs, 12 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, output_size_1, output_size_2])),
    # fully connected, 8*8*12 inputs, 256 outputs
    'wd1': tf.Variable(tf.random_normal([8 * 8 * output_size_2, output_size_3])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([output_size_3, output_size_4]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([output_size_1])),
    'bc2': tf.Variable(tf.random_normal([output_size_2])),
    'bd1': tf.Variable(tf.random_normal([output_size_3])),
    'out': tf.Variable(tf.random_normal([output_size_4]))
}

X, Y, Xv, Yv, Xt, Yt = read_images(image_path, batch_size)

# Construct model
logits = conv_net(X, weights, biases, dropout)
prediction = tf.nn.softmax(conv_net(Xv, weights, biases, dropout=1.0))
prediction_t = tf.nn.softmax(conv_net(Xt, weights, biases, dropout=1.0))

# Define loss and optimizer
with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        # loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    tf.summary.scalar('crocess_entropy', loss_op)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

# Evaluate model
with tf.name_scope('valid_accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Yv, tf.int64))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('valid_accuracy', accuracy)

with tf.name_scope('test_accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred_t = tf.equal(tf.argmax(prediction_t, 1), tf.cast(Yt, tf.int64))
    with tf.name_scope('accuracy'):
        accuracy_t = tf.reduce_mean(tf.cast(correct_pred_t, tf.float32))
tf.summary.scalar('test_accuracy', accuracy_t)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(summary_path)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Run the initializer
    sess.run(init)

    tf.train.start_queue_runners()
    max_acc = 0.0
    acc = 0.0
    for step in range(1, num_steps + 1):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        # sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            summary, _, loss, acc = sess.run([merged, train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))
            writer.add_summary(summary, step)
            if max_acc < acc:
                max_acc = acc
                if max_acc > 0.90:
                    saver.save(sess, './my_tf_model')
                    print("Saved")
        else:
            summary, _ = sess.run([merged, train_op])
            writer.add_summary(summary, step)

    print("Optimization Finished!")

    print("Testing Accuracy:",
          sess.run(accuracy_t))

    coord.request_stop()
    coord.join(threads)

    # Calculate accuracy for 256 MNIST test images
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
    #                                   Y: mnist.test.labels[:256],
    #                                   keep_prob: 1.0}))
