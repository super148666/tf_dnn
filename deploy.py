from __future__ import division, print_function, absolute_import


import sys
import rospy
import cv2
import os
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow as tf
from datetime import datetime

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


model_path = '/home/chao/vision_ws/src/tf_dnn/my_tf_model'
topic_name = '/pylon_camera_node/image_raw'

# Image Parameters
image_width = 32
image_height = 32
image_channel = 3
scale_x = 1.0
scale_y = 1.0
roi_upper = 0.5
roi_lower = 1.0
win_height = 32
win_width = 32
win_stride = 10


# Network Parameters
# num_input = image_width * image_height * image_channel  # MNIST data input (img shape: 28*28)
# num_classes = 2  # MNIST total classes (0-9 digits)
# output_size_1 = 8
# output_size_2 = 16
# output_size_3 = 512
# output_size_4 = num_classes



# # # tf Graph input
# X = tf.placeholder(tf.float32, [None, num_input])
# Y = tf.placeholder(tf.float32, [None, num_classes])


# # Create some wrappers for simplicity
# def conv2d(x, W, b, strides=1):
#     # Conv2D wrapper, with bias and relu activation
    
#     x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
#     x = tf.nn.bias_add(x, b)
#     return tf.nn.tanh(x)


# def maxpool2d(x, k=2):
#     # MaxPool2D wrapper
#     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
#                           padding='SAME')


# def fc1d(x, W, b, act=tf.nn.relu):
#     x_reshaped = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
#     pre_activations = tf.add(tf.matmul(x_reshaped, W), b)
#     activations = act(pre_activations, name='activation')
    
#     return activations


# def dropout1d(x, dropout):
#     dropped = tf.nn.dropout(x, dropout)

#     return dropped

# # Create model
# def conv_net(x, weights, biases, dropout):
#     # Reshape to match picture format [Height x Width x Channel]
#     # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
#     x = tf.reshape(x, shape=[-1, image_height, image_width, image_channel])

#     # Convolution Layer
#     # input size: 32x32x3
#     # output size: 32x32x3
#     conv1 = conv2d(x, weights['wc1'], biases['bc1'])
#     # Max Pooling (down-sampling)
#     # input size: 32x32x3
#     # output size: 16x16x3
#     conv1 = maxpool2d(conv1, k=2)

#     # Convolution Layer
#     # input size: 16x16x3
#     # output size: 16x16x6
#     conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
#     # Max Pooling (down-sampling)
#     # input size: 16x16x6
#     # output size: 8x8x6
#     conv2 = maxpool2d(conv2, k=2)

#     # Fully connected layer
#     # input size: 8x8x6
#     # output size: 1024
#     # Reshape conv2 output to fit fully connected layer input
#     fc1 = fc1d(conv2, weights['wd1'], biases['bd1'], act=tf.nn.tanh)
    
#     # Dropout layer
#     dp1 = dropout1d(fc1, dropout)

#     # Output, class prediction
#     # input size: 1024
#     # output size: 2
#     out = fc1d(dp1, weights['out'], biases['out'], act=tf.identity)
#     return out


# # Store layers weight & bias
# weights = {
#     # 5x5 conv, 3 input, 6 outputs
#     'wc1': tf.Variable(tf.random_normal([3, 3, image_channel, output_size_1])),
#     # 5x5 conv, 6 inputs, 12 outputs
#     'wc2': tf.Variable(tf.random_normal([3, 3, output_size_1, output_size_2])),
#     # fully connected, 8*8*12 inputs, 256 outputs
#     'wd1': tf.Variable(tf.random_normal([8 * 8 * output_size_2, output_size_3])),
#     # 1024 inputs, 10 outputs (class prediction)
#     'out': tf.Variable(tf.random_normal([output_size_3, output_size_4]))
# }

# biases = {
#     'bc1': tf.Variable(tf.random_normal([output_size_1])),
#     'bc2': tf.Variable(tf.random_normal([output_size_2])),
#     'bd1': tf.Variable(tf.random_normal([output_size_3])),
#     'out': tf.Variable(tf.random_normal([output_size_4]))
# }


class image_converter:

    def __init__(self, sess, topic_name, output_tensor, roi_upper=0.5, roi_lower=1.0, win_width=32, win_height=32, win_stride=8, scale_x=0.6, scale_y=0.6):
        self.topic_active = False
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic_name,Image,self.callback)
        self.raw_image = None
        self.proc_image = None
        self.image_width = None
        self.image_height = None
        self.image_channel = None
        self.proc_width = None
        self.proc_height = None
        self.proc_channel = None
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.win_height = win_height
        self.win_width = win_width
        self.win_stride = win_stride
        self.windows = None
        self.windows_start_points = None
        
        # wait for first rostopic
        while self.topic_active == False:
            pass
        print("ROS topic: %s" % topic_name + " is active now")
        
        # update image shape
        print("raw image shape: ", self.raw_image.shape)
        (self.image_height, self.image_width, self.image_channel) = self.raw_image.shape
        print("proc image shape: ", self.proc_image.shape)
        (self.proc_height, self.proc_width, self.proc_channel) = self.proc_image.shape
        

        # check roi setting
        if roi_upper>roi_lower or roi_upper>=1.0 or roi_lower<=0.0:
            print("invalid roi setting ")
            exit()
        
        # setup roi
        self.roi_upper = (int)(roi_upper*self.proc_height)
        self.roi_lower = (int)(roi_lower*self.proc_height)

        # setup tensorflow model
        self.out = output_tensor
        self.results = None
        
        self.sess = sess
                      
    def callback(self,data):
        self.init = tf.global_variables_initializer()
        try:
            self.raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # resize image
        self.proc_image=cv2.resize(self.raw_image, (0,0), fx=self.scale_x, fy=self.scale_y)
        # print("resized image: ", self.proc_image.shape)

        

        # cv2.imshow("Image window", self.raw_image)
        # cv2.waitKey(1)

        if self.topic_active == False:
            self.topic_active = True
        else:
            self.windows, self.windows_start_points = self.get_image_patch()
            print(self.windows.shape)
            self.results = self.sess.run(self.out, feed_dict={X: self.windows})
            for index in range(0,self.results.shape[0]):
                if self.results[index,0] < self.results[index,1] and self.results[index,1] >0.5 and self.results[index,0] <0.0001:
                # if self.results[index,1] < self.results[index,0] and self.results[index,0] >0.95:
                    cv2.rectangle(self.proc_image, (self.windows_start_points[index][0], self.windows_start_points[index][1]), (self.windows_start_points[index][0]+self.win_height, self.windows_start_points[index][1]+self.win_width), (0,0,255), 1) 
                    
        cv2.imshow("proc iamge", self.proc_image)
        cv2.waitKey(1)


    def get_image_patch(self):
        images = list()
        start_points = list()
        y = self.roi_upper
        while y < (self.roi_lower - self.win_height):
            x = 0
            while x < (self.proc_width - self.win_width):
                image = self.proc_image[y:y+self.win_height, x:x+self.win_width]
                images.append(image)
                start_points.append(np.array([x, y]))
                x = x + self.win_stride
            y = y + self.win_stride
        
        # print("number of windows: ", len(start_points))
        images_array = np.asarray(images)
        # print("shape of images array: ", images_array.shape)
        images_array = np.reshape(images_array, [-1, image_height, image_width, image_channel])
        # print("shape of flatten array: ", images_array.shape)
        
        return images_array, start_points
            

tf.reset_default_graph()

 

# prediction = tf.nn.softmax(conv_net(X, weights, biases, dropout=1.0))
# prediction = conv_net(X, weights, biases, dropout=1.0)

# init = tf.global_variables_initializer()

# saver = tf.train.Saver()

# sess = tf.Session()

# load_path = saver.restore(sess, model_path)
# print("Model restored from files: %s" % load_path)
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.import_meta_graph(model_path+'.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('input:0')
prediction = tf.get_collection('prediction')[0]

rospy.init_node('image_converter', anonymous=True)

ic = image_converter(sess,topic_name,prediction,scale_x=scale_x,scale_y=scale_y,win_stride=win_stride,win_height=win_height,win_width=win_width,roi_lower=roi_lower,roi_upper=roi_upper)

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down")
cv2.destroyAllWindows()
coord.request_stop()
coord.join(threads)