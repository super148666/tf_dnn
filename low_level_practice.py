from __future__ import division, print_function, absolute_import

import tensorflow as tf

import os
import glob
import random
import tempfile
import time
import imageio
import numpy as np
from datetime import datetime
from scipy import stats
import cv2

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


image_path_1 = [
                # "/media/chao/RAID1_L/chaoz/cone_detection_data/basler32/",
                "/media/chao/RAID1_L/chaoz/cone_detection_data/basler_new/",
                # "/media/chao/RAID1_L/chaoz/cone_detection_data/small_dataset/",
                # "/media/chao/RAID1_L/chaoz/cone_detection_data/basler/",
                "/media/chao/RAID1_L/chaoz/cone_detection_data/webcam/"
                ]

image_path_2 = [
                # "/media/chao/RAID1_L/chaoz/cone_detection_data/basler32/",
                # "/media/chao/RAID1_L/chaoz/cone_detection_data/basler_new/",
                # "/media/chao/RAID1_L/chaoz/cone_detection_data/small_dataset/",
                # "/media/chao/RAID1_L/chaoz/cone_detection_data/basler/",
                # "/media/chao/RAID1_L/chaoz/cone_detection_data/webcam/"
                ]

current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

summary_path = '/media/chao/RAID1_L/chaoz/tf_summary/' + current_time_str + '/'

# model_path = '/media/chao/RAID1_L/chaoz/tf_model/' + current_time_str + '/'
model_path = '/media/chao/RAID1_L/chaoz/tf_model/latest/model'


# Image Parameters
image_width = 32
image_height = 32
image_channel = 3

# Dataset Partition
train_partition = 0.8

# Training Parameters
samples_each_class = 100000
learning_rate = 0.0001
decay_rate = 0.6
num_steps = 10000
num_epochs = 100
batch_size = 256
display_step = 10


# Network Parameter
num_input = image_width * image_height * image_channel  # MNIST data input (img shape: 28*28)
num_classes = 2  # MNIST total classes (0-9 digits)
dropout = 0.0  # Dropout, probability to keep units
conv1_size = 3
wc1_size = 18
pool1_size = 2
conv2_size = 3
wc2_size = 36
pool2_size = 2

wd1_size = 1024
output_size_1 = wc1_size
output_size_2 = wc2_size
output_size_3 = wd1_size
output_size_4 = num_classes

# # tf Graph input
X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel], name='input')
Y = tf.placeholder(tf.int32, [None], name='label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)


# Reading the dataset
# output as np array with image, label
def read_images(dataset_paths, split_ratio = 1.0, image_channel=3):
    images_1, labels_1, images_2, labels_2 = list(), list(), list(), list()
    
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
            images_temp, labels_temp = list(), list()
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
                if sample.endswith('.png'):
                    count += 1
            split_index = int(count*split_ratio) 
            print(split_index,'/',count)
            count = 0
            for sample in walk[2]:
                # Only keeps png images
                if sample.endswith('.png'):
                    # print(count,'/',split_index)
                    image = imageio.imread(os.path.join(c_dir,sample))
                    image = cv2.resize(image, (32,32))
                    # if count == 0:
                        # cv2.imshow('samples',image)
                        # cv2.waitKey(1)
                    if count < split_index:
                        images_1.append(image)
                        labels_1.append(label)
                    else:
                        images_2.append(image)
                        labels_2.append(label)
                    count += 1

    seed = datetime.now()
    random.Random(seed).shuffle(labels_1)
    random.Random(seed).shuffle(images_1)
    seed = datetime.now()
    random.Random(seed).shuffle(labels_2)
    random.Random(seed).shuffle(images_2)

    image_1 = np.asarray(images_1)
    label_1 = np.asarray(labels_1)
    image_2 = np.asarray(images_2)
    label_2 = np.asarray(labels_2)

    return image_1, label_1, image_2, label_2


# Preprocessing input data
# reshape np array
# standardisation
def preprocessing(input_images, image_height=32, image_width=32, image_channel=3, standardisation=True):
    output_images = np.reshape(input_images, [-1, image_height, image_width, image_channel])
    output_images = output_images / 127.5 - 1.0
    # if standardisation:
    #     for index in range(0,output_images.shape[0]):
    #         output_images[index] = stats.zscore(output_images[index])
    print(output_images.shape)

    return output_images



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
            # for input_index in range(0,W.get_shape()[2]):
            #     for output_index in range(0,W.get_shape[3]):
                    
                    
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
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], layer_name='conv1',act=tf.nn.leaky_relu)
    # Batch Normalization
    mean1, var1 = tf.nn.moments(conv1, [0])
    conv1 = tf.nn.batch_normalization(conv1,mean1,var1,shift1,scale1,1e-8)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=pool1_size, layer_name='max_pool1')

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], layer_name='conv2',act=tf.nn.leaky_relu)
    mean2, var2 = tf.nn.moments(conv2, [0])
    conv2 = tf.nn.batch_normalization(conv2,mean2,var2,shift2,scale2,1e-8)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=pool2_size, layer_name='max_pool2')

    # Fully connected layer
    fc1 = fc1d(conv2, weights['wd1'], biases['bd1'], layer_name='fc1', act=tf.nn.leaky_relu)
    
    # Dropout layer
    dp1 = dropout1d(fc1, dropout, 'dp1')

    # Output, class prediction
    out = fc1d(dp1, weights['out'], biases['out'], layer_name='out', act=tf.identity)


    return out


with tf.Session() as sess:
    global_step = tf.Variable(0, trainable=False, name='global_step')
    scale1 = tf.Variable(tf.ones([32,32,output_size_1]))
    shift1 = tf.Variable(tf.zeros([32,32,output_size_1]))
    scale2 = tf.Variable(tf.ones([16,16,output_size_2]))
    shift2 = tf.Variable(tf.zeros([16,16,output_size_2]))
    # Store layers weight & bias
    weights = {
        
        'wc1': tf.Variable(tf.random_normal([conv1_size, conv1_size, image_channel, output_size_1])),
        
        'wc2': tf.Variable(tf.random_normal([conv2_size, conv2_size, output_size_1, output_size_2])),

        'wd1': tf.Variable(tf.random_normal([int(image_height/pool1_size/pool2_size) * int(image_width/pool1_size/pool2_size) * output_size_2, output_size_3])),

        'out': tf.Variable(tf.random_normal([output_size_3, output_size_4]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([output_size_1])),
        'bc2': tf.Variable(tf.random_normal([output_size_2])),
        'bd1': tf.Variable(tf.random_normal([output_size_3])),
        'out': tf.Variable(tf.random_normal([output_size_4]))
    }



    # Load image from folder paths
    train_images, train_labels, valid_images, valid_labels = read_images(image_path_1,split_ratio=train_partition)
    # valid_images, valid_labels, _, _ = read_images(image_path_1)
    # print(train_images.shape, train_labels.shape)
    train_size = len(train_labels)
    valid_size = len(valid_labels)
    
    # Preprocess images - resize and standardization
    train_images = preprocessing(train_images)
    valid_images = preprocessing(valid_images)


    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    # ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds = ds.batch(batch_size)
    print(ds.output_types)
    print(ds.output_shapes)
    ds = ds.repeat(num_epochs)
    ds = ds.shuffle(train_size)
    # ds = ds.repeat()

    itr = ds.make_initializable_iterator()

    # Print debug info
    print("=====DATASET INFO")
    print("train set size: ", train_size)
    print("valid set size: ", valid_size)
    print("batch size: ", batch_size)
    print("image size: "+str(image_height)+'x'+str(image_width)+'x'+str(image_channel))

    print("=====TRAINING INFO")
    print("learning rate: ",learning_rate)
    print("batch size: ", batch_size)
    print("num of epochs: ", num_steps)
    print("record summary each " + str(display_step) + " steps")


    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(conv_net(X, weights, biases, dropout=1.0))
    steps_per_epoch = int(train_size/batch_size)
    decay_lr = tf.train.exponential_decay(learning_rate, tf.train.get_global_step(), steps_per_epoch, decay_rate, staircase=True)

    # Define loss and optimizer
    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=Y))
        tf.summary.scalar('crocess_entropy', loss_op)

    with tf.name_scope('train'):

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#, epsilon=0.1)
        train_op = optimizer.minimize(loss_op, global_step=global_step)

    # Evaluate model
    with tf.name_scope('valid_accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Y, tf.int64))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('valid_accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    for filename in glob.glob(model_path+'*'):
        os.remove(filename)

    saver = tf.train.Saver(max_to_keep=10)
    saver_fix_steps = tf.train.Saver(max_to_keep=0)
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
    print("Start Training:")
    epoch = 0
    step = 0
    
    sess.run(itr.initializer, feed_dict={X: train_images, Y: train_labels})
    for epoch in range(1,num_epochs+1):
        
        # sess.run(itr.initializer)
        while step < steps_per_epoch:
            try:
                batch_x, batch_y = sess.run(itr.get_next())
            except tf.errors.OutOfRangeError:
                break
            step += 1
            sess.run(train_op,feed_dict={X: batch_x, Y: batch_y, keep_prob: (1 - dropout)})

            if step % display_step == 0 or step == 1 or step == steps_per_epoch:
                
                # Calculate batch loss and accuracy
                # summary, _, loss, acc = sess.run([merged, train_op, loss_op, accuracy],feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout},options=options, run_metadata=run_metadata)
                summary, loss, acc = sess.run([merged, loss_op, accuracy],feed_dict={X: valid_images, Y: valid_labels, keep_prob: 1.0})
                
                # print(decay_lr.eval())
                g_step = sess.run(tf.train.get_global_step())
                print("Epoch "+str(epoch)+", Step " + str(step)+'/'+str(steps_per_epoch) + ", Minibatch Loss= " +
                        "{:.4f}".format(loss) + ", Training Accuracy= " +
                        "{:.5f}".format(acc))
                # writer.add_run_metadata(run_metadata, 'step%03d' % step)
                
                writer.add_summary(summary, g_step)
                if step == steps_per_epoch and acc > 0.98:
                    saver_fix_steps.save(sess,model_path, global_step=g_step)

                if max_acc <= acc:
                    max_acc = acc
                    
                    if max_acc > 0.98:
                        saver.save(sess, model_path+'-'+str(epoch)+'-'+str(step)+"-{:.3f}".format(max_acc))
                        print("Saved")
        
        step = 0

        print('epoch finished: No.',epoch)


    print("Optimization Finished!")

    # print("Testing Accuracy:",
    # sess.run(accuracy_t))

    coord.request_stop()
    coord.join(threads)