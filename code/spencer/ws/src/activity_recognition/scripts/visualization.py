import numpy
import roslib;
import rospy;
import math;
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
import time

from dataset import DatasetPerson


LINKS = {'torso' : ['neck', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
         'neck' : ['head'], 
         'left_shoulder' : ['left_elbow'],
         'right_shoulder' : ['right_elbow', 'left_shoulder'],
           'right_elbow' : ['right_hand'], 
           'left_elbow' : ['left_hand'], 
           'left_hip' : ['left_knee', 'right_hip'], 
           'right_hip' : ['right_knee'],
           'left_knee' : ['left_foot'], 
           'right_knee' : ['right_foot'],}



JOINTS_WITH_ORIENTATION = ['head', 'neck', 'torso', 'left_shoulder', 'left_elbow', 
                             'right_shoulder', 'right_elbow', 'left_hip', 'left_knee',
                             'right_hip', 'right_knee']

JOINTS_WITHOUT_ORIENTATION = ['left_hand', 'right_hand', 'left_foot', 'right_foot']

JOINTS = JOINTS_WITH_ORIENTATION + JOINTS_WITHOUT_ORIENTATION



class Joint:
  position = None;
  orientation = None;
    
  def __str__(self):
    return "Joint[\n Position: %s,\n Orientation:\n %s ]" % (self.position, self.orientation)
      

def parse_joint(data):
  joint = Joint();
  joint.position = numpy.array(data)
  joint.position = numpy.array(data)
  return joint
  

class Pose:
  joints = dict();
   
  def __init__(self, data):
    joint_number = 0;

    for joint_name in JOINTS_WITH_ORIENTATION:
      joint = parse_joint(data[joint_number*3:joint_number*3+3]);
      joint_number += 1;
      self.joints[joint_name] = joint;

    for joint_name in JOINTS_WITHOUT_ORIENTATION:
      joint = parse_joint(data[joint_number*3:joint_number*3+3]);
      joint_number += 1;
      self.joints[joint_name]  = joint;




topic = 'visualization_marker_array'
publisher = rospy.Publisher(topic, MarkerArray)



def create_joint_message(joint, id=0):  
  marker = Marker()
  marker.header.frame_id = "/skeleton"
  marker.type = marker.SPHERE
  marker.id = id
  marker.action = marker.ADD
  marker.pose.position.x = joint.position[0]
  marker.pose.position.y = joint.position[1]
  marker.pose.position.z = joint.position[2]
  marker.scale.x = 20.0
  marker.scale.y = 20.0
  marker.scale.z = 20.0
  marker.color.a = 1.0
  marker.color.r = 1.0
  marker.color.g = 1.0
  marker.color.b = 0.0

  return marker

  

def create_link_message(pose, id=0):

  def pos2Point(joint):
    return Point(joint.position[0], joint.position[1], joint.position[2]);

  points = []
  for jointName1 in LINKS.keys():
    for jointName2 in LINKS[jointName1]:
      joint1 = pose.joints[jointName1];
      joint2 = pose.joints[jointName2];
      points.append(pos2Point(joint1));
      points.append(pos2Point(joint2));

  marker = Marker()
  marker.header.frame_id = "/skeleton"
  marker.type = marker.LINE_LIST
  marker.id = id
  marker.action = marker.ADD
  marker.scale.x = 8.0
  marker.color.a = 1.0
  marker.color.r = 1.0
  marker.points = points

  return marker


  
def create_pose_message(pose):
  markerArray = MarkerArray()
  id = 0
  for joint in pose.joints.values():
    markerArray.markers.append(create_joint_message(joint, id))
    id += 1    
    markerArray.markers.append(create_link_message(pose, id))

  return markerArray

def visualize_frame(frame, data):
  publisher.publish(create_pose_message(Pose(data[frame])))



def visualize_interval(data, start_frame=1, end_frame=1000):
  for frame in range(start_frame, end_frame):
    visualize_frame(frame, data);
    time.sleep(1.0/25.0)


def test():
    visualize_interval(DatasetPerson().get_processed_data())



if __name__ == '__main__':
  rospy.init_node('skeleton_pose_visualizer')
