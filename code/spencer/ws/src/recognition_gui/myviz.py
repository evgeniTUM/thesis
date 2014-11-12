#!/usr/bin/env python

import rospy
import std_srvs.srv
import std_msgs.msg

import sys

from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *

import activity_recognition.srv

def call_service(name):
    rospy.ServiceProxy(name, std_srvs.srv.Empty)()


import rviz

class RecognitionGUI(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.frame = rviz.VisualizationFrame()

        self.frame.setSplashPath( "" )

        self.frame.initialize()

        reader = rviz.YamlConfigReader()
        config = rviz.Config()
        reader.readFile( config, "config.rviz" )
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


    def onLoadModelClick( self ):
        call_service('/activity_recognition/load_model')

    def onSaveModelClick( self ):
        call_service('/activity_recognition/save_model')

    def onLoadSamplesClick( self ):
        call_service('/activity_recognition/load_samples')

    def onSaveSamplesClick( self ):
        call_service('/activity_recognition/save_samples')


    def get_class_label(self):
        items = rospy.ServiceProxy('/activity_recognition/get_classes', activity_recognition.srv.Classes)().classes
        
        item, ok = QInputDialog.getItem(self, "QInputDialog.getItem()",
                "Activity:", items, 0, False)
        if ok and item:
            return items.index(item)
        else:
            return -1



    def onRecognitionClick( self ):
        if self.recognition_button.text() == 'Start Recognition':
            self.recognition_button.setText('Stop Recognition')
            self.activity_label.setVisible(True)
            call_service('/activity_recognition/start_recognition')
        else:
            self.recognition_button.setText('Start Recognition')
            self.activity_label.setVisible(False)
            call_service('/activity_recognition/stop_recognition')



    def onRecordingClick( self ):
        if self.recording_button.text() == 'Start Recording':
            self.recording_button.setText('Stop Recording')

            call_service('/activity_recognition/start_recording')
        else:
            self.recording_button.setText('Start Recording')
            rospy.ServiceProxy('/activity_recognition/stop_recording', activity_recognition.srv.Recording)(self.get_class_label())
                    
                
    def onLearnModelClick(self):
        call_service('/activity_recognition/learn_model')

    def status_callback(self, msg):
        self.status.setText(msg.data)
        

    def activity_callback(self, msg):
        self.activity_label.setText(msg.data)
        

if __name__ == '__main__':

    rospy.init_node('activity_recognition_gui')

    app = QApplication( sys.argv )

    gui = RecognitionGUI()
    gui.resize( 500, 500 )
    gui.show()

    rospy.Subscriber('/activity_recognition/status', std_msgs.msg.String, gui.status_callback)
    rospy.Subscriber('/activity_recognition/activity', std_msgs.msg.String, gui.activity_callback)

    app.exec_()

