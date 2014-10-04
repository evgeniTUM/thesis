#!/usr/bin/env python  
import roslib
import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv
import copy
import pickle

import threading

from bof_approach import *
from dataset import label_to_class, convert_to_features

lock = threading.Lock()
poses = []


frames_per_second = 25


# pkl_file = open('samples.pkl', 'rb')
# samples = pickle.load(pkl_file)
# pkl_file.close()


# pkl_file = open('labels.pkl', 'rb')
# labels = pickle.load(pkl_file)
# pkl_file.close()

# svm, kmeans = learn(samples, labels)

svm = pickle.load(open('svm.pkl', 'rb'))
kmeans = pickle.load(open('kmeans.pkl', 'rb'))

def recognize (): 
    global poses
    
    threading.Timer(2.0, recognize).start (); 

    with lock:
        data = copy.copy(poses)
        poses = []
        if len(data) == 0:
            print "No poses"
        else:
            print data[0]
            features = convert_to_features(np.vstack(data))
            print label_to_class(predict(svm, kmeans, features))
            print features.shape



# 1 -> HEAD
# 2 -> NECK
# 3 -> TORSO
# 4 -> LEFT_SHOULDER
# 5 -> LEFT_ELBOW
# 6 -> RIGHT_SHOULDER
# 7 -> RIGHT_ELBOW
# 8 -> LEFT_HIP
# 9 -> LEFT_KNEE
# 10 -> RIGHT_HIP
# 11 -> RIGHT_KNEE
# 12 -> LEFT_HAND
# 13 -> RIGHT_HAND
# 14 -> LEFT_FOOT
# 15 -> RIGHT_FOOT

joints = ['head','neck','torso','left_shoulder','left_elbow',
          'right_shoulder','right_elbow','left_hip','left_knee',
          'right_hip','right_knee','left_hand', 'right_hand',
          'left_foot','right_foot']


if True: # __name__ == '__main__':
    # work only with one person for now
    person_id = 1

    rospy.init_node('activity_recognition')
    listener = tf.TransformListener()

    rate = rospy.Rate(float(frames_per_second))
    recognize()

    print "recognition started"
    while not rospy.is_shutdown():
        try:
            pose = []
            
            for joint in joints:
                (trans,rot) = listener.lookupTransform(
                    joint+'_'+str(person_id), 
                    '/openni_depth_frame', rospy.Time(0))
                pose.append(trans[0]*1000)
                pose.append(trans[1]*1000)
                pose.append(trans[2]*1000)

            with lock:
                poses.append(np.hstack(pose))
                    
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        

        rate.sleep()
