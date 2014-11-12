#!/usr/bin/env python  
import roslib
import rospy
import tf
import geometry_msgs.msg

import visualization_msgs.msg
import std_msgs.msg
import std_msgs.msg
import std_srvs.srv

import activity_recognition.srv

import math
import copy
import pickle
import sys
import threading
import copy


from bof_approach import *
from dataset import label_to_class, class_to_label, class_labels,  convert_to_features
from visualization import Pose, create_pose_message

lock = threading.Lock()
recording_lock = threading.Lock()
recording = False
recognition = False
poses = []
record = []

# samples
samples = []
labels = []

# model
svm = []
kmeans = []


status_publisher = []
activity_publisher = []

frames_per_second = 25


### ---------------------------------------------------
## Visualization
topic = 'visualization_marker_array'
publisher = rospy.Publisher(topic, visualization_msgs.msg.MarkerArray)
### --------------------------------------------------


## ----------------------------------------------------
## Get package path
import rospkg
rospack = rospkg.RosPack()
path = rospack.get_path('activity_recognition')
#------------------------------------------------------


def status(message):
    status_publisher.publish(message)

def set_activity_status(activity):
    activity_publisher.publish(activity)


def learn_model(req=False):
    global svm, kmeans    
    if(len(samples) < 2):
        status("Too few samples for learning")
    else:
        svm, kmeans = learn(samples, labels)
        status("Model learned")

    return std_srvs.srv.EmptyResponse()

def save_model(req=False):
    global svm, means
    pickle.dump(svm, open(path+'/resources/svm.pkl', 'wb'))
    pickle.dump(kmeans, open(path+'/resources/kmeans.pkl', 'wb'))
    status("Model saved")
    return std_srvs.srv.EmptyResponse()



def load_model(req=False):
    global svm, kmeans
    svm = pickle.load(open(path+'/resources/svm.pkl', 'rb'))
    kmeans = pickle.load(open(path+'/resources/kmeans.pkl', 'rb'))
    status("Model loaded")
    return std_srvs.srv.EmptyResponse()


def save_samples(req=False):
    global samples, labels
    pickle.dump(samples, open(path+'/resources/samples.pkl', 'wb'))
    pickle.dump(labels, open(path+'/resources/labels.pkl', 'wb'))
    status("Samples saved")
    return std_srvs.srv.EmptyResponse()


def load_samples(req=False):
    global samples, labels
    samples = pickle.load(open(path+'/resources/samples.pkl', 'rb'))
    labels = pickle.load(open(path+'/resources/labels.pkl', 'rb'))
    status("Samples loaded")
    return std_srvs.srv.EmptyResponse()

def set_recognition(value):
    global recognition, lock
    with lock:
        recognition = value
    return std_srvs.srv.EmptyResponse()

def start_recognition(req=False):
    set_recognition(True)
    status("Recognition started...")


def stop_recognition(req=False):
    set_recognition(False)
    status("Recognition stopped...")
    return std_srvs.srv.EmptyResponse


def recognize (): 
    global poses, recognition, lock
    
    threading.Timer(2.0, recognize).start (); 

    with lock:
        if recognition == False:
            return
 
        data = copy.copy(poses)
        poses = []
        if len(data) == 0:
            set_activity_status("No poses")
        else:
            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # d = np.array(data[0]).reshape((15,3))
            # ax.scatter(d[:,0], d[:,1], d[:,2])

            features = convert_to_features(np.vstack(data))
            set_activity_status(label_to_class(predict(svm, kmeans, features)))



def get_classes(req=False):
    return activity_recognition.srv.ClassesResponse(class_labels)


def callback(pose):
    global poses, record, lock, recording_lock, recognition, recording, publisher
    
    publisher.publish(create_pose_message(Pose(pose.data)))


    with lock:
        if recognition:
            poses.append(pose.data)
    with recording_lock:
        if recording:
            record.append(pose.data)


def start_recording(req=True):
    global recording_lock, recording
    with recording_lock:
        recording = True 
        status("Activity recording started...")


def stop_recording(req):
    global recording_lock, recording
    with recording_lock:
        recording = False
        label = req.class_label
        if label >= 0:
            save_record(label)
            status("Activity record was added to the samples with class: " + class_labels[label])
        else:
            status("Activity recording was canceled")


def save_record(label):
    global recording_lock, samples, labels, record
    samples.append(convert_to_features(np.array(record)))
    labels.append(label)
    record = []



if __name__ == '__main__':

    rospy.init_node('activity_recognition')
    rospy.Subscriber("/openni_tracker/pose", std_msgs.msg.Int32MultiArray, callback)

    ## status and activity messages
    status_publisher = rospy.Publisher("~status", std_msgs.msg.String)
    activity_publisher = rospy.Publisher("~activity", std_msgs.msg.String)

    ## Services
    rospy.Service('~load_model', std_srvs.srv.Empty, load_model)
    rospy.Service('~save_model', std_srvs.srv.Empty, save_model)
    rospy.Service('~load_samples', std_srvs.srv.Empty, load_samples)
    rospy.Service('~save_samples', std_srvs.srv.Empty, save_samples)

    rospy.Service('~learn_model', std_srvs.srv.Empty, learn_model)

    rospy.Service('~start_recognition', std_srvs.srv.Empty, start_recognition)
    rospy.Service('~stop_recognition', std_srvs.srv.Empty, stop_recognition)
    rospy.Service('~start_recording', std_srvs.srv.Empty, start_recording)
    rospy.Service('~stop_recording', activity_recognition.srv.Recording, stop_recording)

    rospy.Service('~get_classes', activity_recognition.srv.Classes, get_classes)



    load_model()

    recognize()

    rospy.spin()
