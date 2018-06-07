from __future__ import division, print_function, absolute_import

import tensorflow as tf

import os
import glob
import random
import tempfile
from datetime import datetime

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

seed = 100

image_path = [
            #   "/media/chao/RAID1_L/chaoz/cone_detection_data/basler32/",
              "/media/chao/RAID1_L/chaoz/cone_detection_data/basler_new/",
            #   "/media/chao/RAID1_L/chaoz/cone_detection_data/basler/",
            #   "/media/chao/RAID1_L/chaoz/cone_detection_data/webcam/"
            ]

current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

summary_path = '/media/chao/RAID1_L/chaoz/tf_summary/' + current_time_str + '/'

# model_path = '/media/chao/RAID1_L/chaoz/tf_model/' + current_time_str + '/'
model_path = '/home/chao/vision_ws/src/tf_dnn/my_tf_model'


# Image Parameters
image_width = 32
image_height = 32
image_channel = 3

# Dataset Partition
train_partition = 0.8
valid_partition = 0.2
test_partition = 1.0 - valid_partition - train_partition

# Training Parameters
samples_each_class = 100000
learning_rate = 0.002
num_steps = 10000
batch_size = 128
display_step = 10


# Network Parameter
num_input = image_width * image_height * image_channel  # MNIST data input (img shape: 28*28)
num_classes = 2  # MNIST total classes (0-9 digits)
dropout = 1.0  # Dropout, probability to keep units
output_size_1 = 9
output_size_2 = 9
output_size_3 = 256
output_size_4 = num_classes

# # tf Graph input
X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel], name='input')
Y = tf.placeholder(tf.int32, [None], name='label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)


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
        # except Exception:  # Python 3
            # classes = sorted(os.walk(dataset_path).__next__()[1])
        except StopIteration:
            pass
        # List each sub-directory (the classes)
        print(classes)

        for c in classes:
            label = (int)(c)
            print(c,label)
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            # except Exception:  # Python 3
                # walk = os.walk(c_dir).__next__()
            except StopIteration:
                pass
            # Add each image to the training set
            count = 0
            random.shuffle(walk[2])
            for sample in walk[2]:
                # Only keeps png images
                if count > samples_each_class:
                    break
                if sample.endswith('.png'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
                    count=count+1

    random.Random(seed).shuffle(labels)
    random.Random(seed).shuffle(imagepaths)
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
    
    # # Normalize

    train_image = tf.image.per_image_standardization(train_image)
    valid_image = tf.image.per_image_standardization(valid_image)
    test_image = tf.image.per_image_standardization(test_image)
    
    # Create batches
    X, Y = tf.train.shuffle_batch([train_image, train_label], batch_size=batch_size,
                          capacity=batch_size * 8, min_after_dequeue=batch_size * 2,
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
#   with tf.name_scope('summaries'):
#     mean = tf.reduce_mean(var)
#     tf.summary.scalar('mean', mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#     tf.summary.scalar('stddev', stddev)
#     tf.summary.scalar('max', tf.reduce_max(var))
#     tf.summary.scalar('min', tf.reduce_min(var))
#     tf.summary.histogram('histogram', var)


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
            # tf.summary.histogram('pre_activations',x)
        activations = act(x, name='activation')
        # tf.summary.histogram('activations', activations)
    return activations


def maxpool2d(x, layer_name, k=2):
    # MaxPool2D wrapper
    with tf.name_scope(layer_name):
        out = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME', name='max_pooling')
        # tf.summary.histogram('max_poolings', out)
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
            # tf.summary.histogram('pre_activations',pre_activations)
        activations = act(pre_activations, name='activation')
        # tf.summary.histogram('activations', activations)

    return activations


def dropout1d(x, dropout, layer_name):
    with tf.name_scope(layer_name):
        # tf.summary.scalar('dropout', dropout)
        dropped = tf.nn.dropout(x, dropout)

    return dropped


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, [-1, image_height, image_width, image_channel])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], layer_name='conv1',act=tf.nn.relu)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2, layer_name='max_pool1')

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], layer_name='conv2',act=tf.nn.tanh)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2, layer_name='max_pool2')

    # Fully connected layer
    fc1 = fc1d(conv2, weights['wd1'], biases['bd1'], layer_name='fc1', act=tf.nn.tanh)
    
    # Dropout layer
    dp1 = dropout1d(fc1, dropout, 'dp1')

    # Output, class prediction
    out = fc1d(dp1, weights['out'], biases['out'], layer_name='out', act=tf.identity)
    return out


with tf.Session() as sess:

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 3 input, 6 outputs
        'wc1': tf.Variable(tf.random_normal([3, 3, image_channel, output_size_1])),
        # 5x5 conv, 6 inputs, 12 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, output_size_1, output_size_2])),
        # fully connected, 8*8*12 inputs, 256 outpprint(classes)uts
        'wd1': tf.Variable(tf.random_normal([8 * 8 * output_size_2, output_size_3])),
        # 1024 inputs, 10 outputs (class predictioprint(classes)n)
        'out': tf.Variable(tf.random_normal([output_size_3, output_size_4]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([output_size_1])),
        'bc2': tf.Variable(tf.random_normal([output_size_2])),
        'bd1': tf.Variable(tf.random_normal([output_size_3])),
        'out': tf.Variable(tf.random_normal([output_size_4]))
    }

    Xa, Ya, Xv, Yv, Xt, Yt = read_images(image_path, batch_size)

    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    prediction_v = tf.nn.softmax(conv_net(Xv, weights, biases, dropout=1.0))

    prediction = tf.nn.softmax(conv_net(X, weights, biases, dropout=1.0))

    # Define loss and optimizer
    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=Y))
            # loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        tf.summary.scalar('crocess_entropy', loss_op)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

    # Evaluate model
    with tf.name_scope('valid_accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_pred = tf.equal(tf.argmax(prediction_v, 1), tf.cast(Yv, tf.int64))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('valid_accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    for filename in glob.glob(model_path+'*'):
        os.remove(filename)

    saver = tf.train.Saver()
    tf.add_to_collection('prediction',prediction)
    # Start training

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    # Run the initializer
    sess.run(init)

    tf.train.start_queue_runners()
    max_acc = 0.0
    acc = 0.0
    for step in range(1, num_steps + 1):

        batch_x , batch_y = sess.run([Xa,Ya])
        # print(batch_x[0])
        # exit()

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            # summary, _, loss, acc = sess.run([merged, train_op, loss_op, accuracy],feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout},options=options, run_metadata=run_metadata)
            summary, _, loss, acc = sess.run([merged, train_op, loss_op, accuracy],feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            print("Step " + str(step) + ", Minibatch Loss= " +
                    "{:.4f}".format(loss) + ", Training Accuracy= " +
                    "{:.5f}".format(acc))
            # writer.add_run_metadata(run_metadata, 'step%03d' % step)
            writer.add_summary(summary, step)
            if step % 500 == 0:
                saver.save(sess, model_path+str(step))

            if max_acc < acc:
                max_acc = acc
                if max_acc > 0.85:
                    saver.save(sess, model_path+"{:.2f}".format(max_acc))
                    print("Saved")
        else:
            sess.run(train_op,feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

    print("Optimization Finished!")

    # print("Testing Accuracy:",
    # sess.run(accuracy_t))

    coord.request_stop()
    coord.join(threads)