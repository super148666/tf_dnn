from __future__ import division, print_function, absolute_import

import tensorflow as tf

import os
import random

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

seed = 100

image_path = [
              "/media/chao/RAID1_L/chaoz/cone_detection_data/basler32/",
              "/media/chao/RAID1_L/chaoz/cone_detection_data/basler/",
              "/media/chao/RAID1_L/chaoz/cone_detection_data/webcam/"
            ]


model_path = '/home/chao/vision_ws/src/tensorflow_dnn/my_tf_model'


# Image Parameters
image_width = 32
image_height = 32
image_channel = 3

# Dataset Partition
train_partition = 0.6
valid_partition = 0.2
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



# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, image_height, image_width, image_channel])

    # Convolution Layer
    # input size: 32x32x3
    # output size: 32x32x3
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    # input size: 32x32x3
    # output size: 16x16x3
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # input size: 16x16x3
    # output size: 16x16x6
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    # input size: 16x16x6
    # output size: 8x8x6
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # input size: 8x8x6
    # output size: 1024
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    # input size: 1024
    # output size: 2
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, image_channel, 6])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 6, 12])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8 * 8 * 12, 256])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([6])),
    'bc2': tf.Variable(tf.random_normal([12])),
    'bd1': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

X, Y, Xv, Yv, Xt, Yt = read_images(image_path, batch_size)

# Construct model
logits = conv_net(X, weights, biases, dropout)
prediction = tf.nn.softmax(conv_net(Xv, weights, biases, dropout=1.0))
prediction_t = tf.nn.softmax(conv_net(Xt, weights, biases, dropout=1.0))

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Yv, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

correct_pred_t = tf.equal(tf.argmax(prediction_t, 1), tf.cast(Yt, tf.int64))
accuracy_t = tf.reduce_mean(tf.cast(correct_pred_t, tf.float32))


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
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))
            if max_acc < acc:
                max_acc = acc
                if max_acc > 0.90:
                    saver.save(sess, './my_tf_model')
                    print("Saved")
        else:
            sess.run(train_op)

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
