#!/usr/bin/env python  
import roslib
import rospy
import math
import tf
import geometry_msgs.msg
import turtlesim.srv
import copy
import pickle
import sys

import threading

from bof_approach import *
from dataset import label_to_class, convert_to_features

lock = threading.Lock()
recording_lock = threading.Lock()
recording = False
recognition = False
poses = []
record = []

samples = []
labels = []

svm = []
kmeans = []

gui = []


frames_per_second = 25



## ----------------------------------------------------
## Get package path

import rospkg
rospack = rospkg.RosPack()
path = rospack.get_path('activity_recognition')
#------------------------------------------------------


def status(message):
    gui.status.setText(message)

def set_activity_status(activity):
    gui.activity_label.setText(activity)


def learn_model(samples, labels):
    svm, kmeans = learn(samples, labels)
    status("Model was learned from the samples")
    return svm, kmeans

def save_model(svm, means):
    pickle.dump(svm, open(path+'/resources/svm.pkl', 'wb'))
    pickle.dump(kmeans, open(path+'/resources/kmeans.pkl', 'wb'))
    status("Model saved")


def load_model():
    svm = pickle.load(open(path+'/resources/svm.pkl', 'rb'))
    kmeans = pickle.load(open(path+'/resources/kmeans.pkl', 'rb'))
    status("Model loaded")
    return svm, kmeans


def save_samples(samples, labels):
    pickle.dump(samples, open(path+'/resources/samples.pkl', 'wb'))
    pickle.dump(labels, open(path+'/resources/labels.pkl', 'wb'))
    status("Samples saved")


def load_samples():
    samples = pickle.load(open(path+'/resources/samples.pkl', 'rb'))
    labels = pickle.load(open(path+'/resources/labels.pkl', 'rb'))
    status("Samples loaded")
    return samples, labels


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




from std_msgs.msg import Int32MultiArray

def callback(pose):
    global poses, record, lock, recording_lock, recognition, recording
    
    with lock:
        if recognition:
            poses.append(pose.data)

    with recording_lock:
        if recording:
            record.append(pose.data)



def save_record(label):
    global recording_lock, samples, labels, record
    samples.append(convert_to_features(record))
    labels.append(label)
    record = []










        

from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *


import rviz

class RecognitionGUI(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.frame = rviz.VisualizationFrame()

        self.frame.setSplashPath( "" )

        self.frame.initialize()

        reader = rviz.YamlConfigReader()
        config = rviz.Config()
        reader.readFile( config, "config.myviz" )
        self.frame.load( config )

        self.setWindowTitle( config.mapGetChild( "Title" ).getValue() )

        # self.frame.setMenuBar( None )
        # self.frame.setStatusBar( None )
        # self.frame.setHideButtonVisibility( False )

        self.manager = self.frame.getManager()

        self.grid_display = self.manager.getRootDisplayGroup().getDisplayAt( 0 )
        
        layout = QVBoxLayout()
        layout.addWidget( self.frame )
        
        # thickness_slider = QSlider( Qt.Horizontal )
        # thickness_slider.setTracking( True )
        # thickness_slider.setMinimum( 1 )
        # thickness_slider.setMaximum( 1000 )
        # thickness_slider.valueChanged.connect( self.onThicknessSliderChanged )
        # layout.addWidget( thickness_slider )


        label = QLabel("Status")
        self.status = label
        layout.addWidget(label)

        activity_label = QLabel("")
        font = QFont()
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(50)
        activity_label.setFont(font)
        activity_label.setVisible(False)
        self.activity_label = activity_label
        layout.addWidget(activity_label)

        
        h_layout = QHBoxLayout()
        
        save_model_button = QPushButton( "Save Model" )
        save_model_button.clicked.connect( self.onSaveModelClick )
        h_layout.addWidget(save_model_button)
        
        load_model_button = QPushButton( "Load Model" )
        load_model_button.clicked.connect( self.onLoadModelClick )
        h_layout.addWidget(load_model_button)

        layout.addLayout(h_layout)


        h2_layout = QHBoxLayout()
        
        save_samples_button = QPushButton( "Save Samples" )
        save_samples_button.clicked.connect( self.onSaveSamplesClick )
        h2_layout.addWidget( save_samples_button )
        
        load_samples_button = QPushButton( "Load Samples" )
        load_samples_button.clicked.connect( self.onLoadSamplesClick )
        h2_layout.addWidget( load_samples_button )

        layout.addLayout(h2_layout)



        command_layout = QHBoxLayout()
        
        recognition_button = QPushButton( "Start Recognition" )
        recognition_button.clicked.connect( self.onRecognitionClick )
        self.recognition_button = recognition_button
        command_layout.addWidget(recognition_button)
        
        recording_button = QPushButton( "Start Recording" )
        recording_button.clicked.connect( self.onRecordingClick )
        self.recording_button = recording_button
        command_layout.addWidget(recording_button)

        layout.addLayout(command_layout)

        learn_model_button = QPushButton( "Learn Model from samples" )
        learn_model_button.clicked.connect( self.onLearnModelClick )
        layout.addWidget(learn_model_button)

        
        
        self.setLayout(layout)

    def onThicknessSliderChanged( self, new_value ):
        if self.grid_display != None:
            self.grid_display.subProp( "Line Style" ).subProp( "Line Width" ).setValue( new_value / 1000.0 )
            self.label.setText(str(new_value))

    def onLoadModelClick( self ):
        global svm, kmeans
        svm, kmeans = load_model()

    def onSaveModelClick( self ):
        global svm, kmeans
        save_model(svm, kmeans)

    def onLoadSamplesClick( self ):
        global samples, labels
        samples, labels = load_samples()

    def onSaveSamplesClick( self ):
        global samples, labels
        save_samples(samples, labels)


    def get_class_label(self):
        items = ("Still", "Running", "Calling", "Eating")
  
        item, ok = QInputDialog.getItem(self, "QInputDialog.getItem()",
                "Activity:", items, 0, False)
        if ok and item:
            return items.index(item)
        else:
            return -1



    def onRecognitionClick( self ):
        global lock, recognition
        with lock:
            if self.recognition_button.text() == 'Start Recognition':
                self.recognition_button.setText('Stop Recogntion')
                recognition = True
                status("Recognition started...")
                self.activity_label.setVisible(True)
            else:
                self.recognition_button.setText('Start Recognition')
                recognition = False
                status("Recognition stopped...")
                self.activity_label.setVisible(False)


    def onRecordingClick( self ):
        global recording_lock, recording
        with recording_lock:
            if self.recording_button.text() == 'Start Recording':
                self.recording_button.setText('Stop Recording')
                recording = True
                status("Activity recording started...")
            else:
                self.recording_button.setText('Start Recording')
                recording = False
                label = self.get_class_label()
                if label >= 0:
                    save_record(label)
                    status("Activity record was added to the samples")
                else:
                    status("Activity recording was canceled")
        
                
    def onLearnModelClick(self):
        global svm, kmeans, samples, labels
        if(len(samples) < 2):
            status("Too few samples for learning")
        else:
            svm, kmeans = learn_model(samples, labels)
            status("Model learned")

             
        

if __name__ == '__main__':

    rospy.init_node('activity_recognition')
    rospy.Subscriber("/openni_tracker/pose", Int32MultiArray, callback)

    svm, kmeans = load_model()

    recognize()
    # rospy.spin()

    app = QApplication( sys.argv )

    gui = RecognitionGUI()
    gui.resize( 500, 500 )
    gui.show()

    app.exec_()


        
        
    
