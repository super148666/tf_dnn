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
from scipy import stats
from datetime import datetime

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


model_path = '/home/chao/vision_ws/src/tf_dnn/my_tf_model0.971'
topic_name = '/pylon_camera_node/image_raw'

# Image Parameters
image_width = 32
image_height = 32
image_channel = 3
scale_x = 1.0
scale_y = 1.0
roi_upper = 0.5
roi_lower = 0.8
win_height = 32
win_width = 32
win_stride = 6



class image_converter:

    def __init__(self, sess, topic_name, output_tensor, roi_upper=0.5, roi_lower=1.0, win_width=32, win_height=32, win_stride=8, scale_x=0.6, scale_y=0.6):
        self.topic_active = False
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic_name,Image,self.callback)
        self.raw_image = None
        self.proc_image = None
        self.disp_image = None
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
        self.feature_detector = cv2.ORB_create()

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
        self.disp_image = cv2.resize(self.raw_image, (0,0), fx=self.scale_x, fy=self.scale_y)

        # self.disp_image = cv2.GaussianBlur(self.disp_image,(3,3),0)
        # self.proc_image = cv2.GaussianBlur(self.disp_image,(5,5),0)
        self.proc_image = self.disp_image / 127.5 - 1.0

        if self.topic_active == False:
            self.topic_active = True
        else:
            
            self.windows, self.windows_start_points = self.get_image_patch()
            kp = self.feature_detector.detect(self.disp_image,None)
            cv2.drawKeypoints(self.disp_image, kp, self.disp_image, color=(255,0,0))
            self.results = self.sess.run(self.out, feed_dict={X: self.windows})
            for index in range(0,self.results.shape[0]):
                # print(self.results[index])
                # if self.results[index,1] >0.00000000 and self.results[index,0] <1.0:
                # if self.results[index,1] < self.results[index,0] and self.results[index,0] >0.95:
                if self.results[index,1]>0.8:
                    # print("cone: ",self.results[index,1])
                    cv2.rectangle(self.disp_image, (self.windows_start_points[index][0], self.windows_start_points[index][1]), (self.windows_start_points[index][0]+self.win_height, self.windows_start_points[index][1]+self.win_width), (0,0,255), 1) 
                    
        
        cv2.imshow("proc iamge", self.disp_image)
        cv2.waitKey(1)


    def get_image_patch(self):
        images = list()
        start_points = list()
        y = self.roi_upper
        while y < (self.roi_lower - self.win_height):
            x = 0
            while x < (self.proc_width - self.win_width):
                image = self.proc_image[y:y+self.win_height, x:x+self.win_width]
                image = stats.zscore(image)
                images.append(image)
                start_points.append(np.array([x, y]))
                x = x + self.win_stride
            y = y + self.win_stride
        
        images_array = np.asarray(images)
        images_array = np.reshape(images_array, [-1, image_height, image_width, image_channel])
        
        return images_array, start_points
            

tf.reset_default_graph()

 
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.import_meta_graph(model_path+'.meta')
saver.restore(sess, model_path)
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