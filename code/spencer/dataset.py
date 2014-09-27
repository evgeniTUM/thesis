import glob
import os
import numpy as np


data_set_indices = []
# indices of positions of first 11 joints (joints with orientation)
# 9 ori + 1 conf   +   3 pos + 1 conf = 14 
for joint in range(0,11):
  for x in range(10,13):
    data_set_indices.append(1 + joint*14 + x);

# indices of hands and feet (no orientation)
for joint in range(0,4):
  for x in range(0,3):
    data_set_indices.append(155 + joint*4 + x);
        

default_data_dir=os.getenv("HOME")+'/data/human_activities'

      
class DatasetPerson:

  data_dir = "";
  person = -1;
  direcotory = "";
  activity_label = dict();
  classes = list();
  activity = ''
  data = None

  def __init__(self, data_dir=default_data_dir, person=1):
    self.data_dir = data_dir;
    self.person = person;
    self.directory = data_dir + '/data'+ str(person) + '/';

    # read labels
    with open(self.directory + '/activityLabel.txt') as f:
      self.activity_label = dict([filter(None, x.rstrip().split(',')) for x in f if x != 'END\n']);

    self.classes = list(set(self.activity_label.values()));
    self.activity = self.activity_label.keys()[0]
    self.load_activity(self.activity)


  def load_activity(self, activity):
    self.activity = activity
    file_name = self.directory + activity + '.txt';
    self.data = np.genfromtxt(file_name, delimiter=',', skip_footer=1);

  def get_processed_data(self):
    data = self.data[:, data_set_indices];

    # take relative position of the joints (rel. to torso)
    for row in data:
      torso_position = row[6:9]
      for joint in range(0, 15):
        row[joint*3:joint*3+3] -= torso_position

    return data

  def get_pose(self, frame):
    return Pose(self.data[frame])
