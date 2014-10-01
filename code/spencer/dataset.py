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

labels = []
labels.append('H')
labels.append('N')
labels.append('T')
labels.append('LS')
labels.append('LE')
labels.append('RS')
labels.append('RE')
labels.append('LHi')
labels.append('LK')
labels.append('RHi')
labels.append('RK')
labels.append('LHa')
labels.append('RHa')
labels.append('LF')
labels.append('RF')




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

    def calc_rotation_matrix(pose):
      torso = pose[6:9]
      right_hip = pose[9*3:9*3+3]
      left_hip = pose[7*3:7*3+3]

      u1 = right_hip - torso
      u2 = left_hip - torso

      u1 = u1 / np.linalg.norm(u1)
      u2 = u2 / np.linalg.norm(u2)

      u3 = np.cross(u1, u2)

      return np.vstack((u1,u2,u3)).T
      


    # take relative position of the joints (rel. to torso)
    for row in data:
      R = calc_rotation_matrix(row)

      torso_position = row[6:9].copy()
      neck_position = row[3:6]

      normalizer = np.linalg.norm(neck_position - torso_position)


      # translate, rotate point and normalize point cloud
      for joint in range(0,15):
        point3d = row[joint*3:joint*3+3]
        point3d = point3d - torso_position
        point3d = np.dot(R, point3d.T)

        row[joint*3:joint*3+3] = point3d/normalizer


    return data

  def get_pose(self, frame):
    return Pose(self.data[frame])

  def plot_pose(self, frame, show_labels=False):
    pose = self.get_processed_data()[frame]
    pose = pose.reshape(15,3)

    print pose

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import proj3d


    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pose[:,0], pose[:,1], pose[:,2])

    if not show_labels:
      return

    label = {}
    for text, x, y, z in zip(labels, pose[:,0], pose[:,1], pose[:,2]):
      x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())

      label[text] = pylab.annotate(
        text, 
        xy = (x2, y2), xytext = (-5, 5),
        textcoords = 'offset points', ha = 'right', va = 'bottom')

      def update_position(e):
        for text, x, y, z in zip(labels, pose[:,0], pose[:,1], pose[:,2]):
          x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
          label[text].xy = x2,y2
          label[text].update_positions(fig.canvas.renderer)
        fig.canvas.draw()
        fig.canvas.mpl_connect('button_release_event', update_position)
        pylab.show()

    fig.canvas.mpl_connect('button_release_event', update_position)
    pylab.show()
      

