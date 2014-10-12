import glob
import os
import numpy as np
import scipy.ndimage.filters



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

joint_labels = []
joint_labels.append('H')
joint_labels.append('N')
joint_labels.append('T')
joint_labels.append('LS')
joint_labels.append('LE')
joint_labels.append('RS')
joint_labels.append('RE')
joint_labels.append('LHi')
joint_labels.append('LK')
joint_labels.append('RHi')
joint_labels.append('RK')
joint_labels.append('LHa')
joint_labels.append('RHa')
joint_labels.append('LF')
joint_labels.append('RF')


class_labels = {
'still': 0,
'talking on the phone': 1,
'writing on whiteboard': 2,
'drinking water': 3,
'rinsing mouth with water': 4,
'brushing teeth': 5,
'wearing contact lenses': 6,
'talking on couch': 7,
'relaxing on couch': 8,
'cooking (chopping)': 9,
'cooking (stirring)': 10,
'opening pill container': 11,
'working on computer': 12,
'random': 13}


def label_to_class(l):
  for activity, label in class_labels.iteritems():
    if label == l:
        return activity


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
        point3d = np.dot(R, point3d.T)
        point3d = point3d - torso_position

        row[joint*3:joint*3+3] = point3d #/normalizer


    return data


 
  def get_features(self, sigma=None):
    data = self.data[:, data_set_indices]

    if sigma is not None:
      for i in range(data.shape[1]):
        data[:, i] = scipy.ndimage.gaussian_filter1d(data[:, i], sigma)
    return convert_to_features(data)

  def get_raw_data(self):
    return self.data[:, data_set_indices]



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
    for text, x, y, z in zip(joint_labels, pose[:,0], pose[:,1], pose[:,2]):
      x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())

      label[text] = pylab.annotate(
        text, 
        xy = (x2, y2), xytext = (-5, 5),
        textcoords = 'offset points', ha = 'right', va = 'bottom')

      def update_position(e):
        for text, x, y, z in zip(joint_labels, pose[:,0], pose[:,1], pose[:,2]):
          x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
          label[text].xy = x2,y2
          label[text].update_positions(fig.canvas.renderer)
        fig.canvas.draw()
        fig.canvas.mpl_connect('button_release_event', update_position)
        pylab.show()

    fig.canvas.mpl_connect('button_release_event', update_position)
    pylab.show()
      



def joint3d(row, i):
  return row[i*3:i*3+3]


def convert_to_features(data):
  
  features = []
  for t in range(1, data.shape[0]):
    feature = []
    for i in range(0,15):
      feature.append(joint3d(data[t], i) - joint3d(data[t-1], i))
     
    for i in range(0,15):
      for j in range(i+1,15):
      #for j in [2, 7, 9]:
        feature.append(joint3d(data[t], i) - joint3d(data[t], j))

    features.append(np.array(feature).flatten())

  return np.array(features)


def read_persons(persons = [1,2,3], sigma=None):
    global classes

    print "Reading persons", persons

    person = []
    for i in persons:
        person.append(DatasetPerson(person=i))

    samples = []
    labels = []

    for personId in range(len(person)):
        for activity, label in person[personId].activity_label.iteritems():
            if label == 'random':
                continue

            person[personId].load_activity(activity)
            sample_features = person[personId].get_features(sigma=sigma)

            
            samples.append(sample_features)
            labels.append(class_labels[label])

    return samples, labels
