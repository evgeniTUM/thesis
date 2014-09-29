import GPy
import numpy as np
import scipy.spatial

from utils import dtw
from dataset import DatasetPerson




def get_mocap_data(subject=35 , motion=01):
    data = []
    seq_index = [0]
    class_index = [0]
    index = 0

    # walk sequences
    for i in range(1):
        data.append(GPy.util.datasets.cmu_mocap(subject, [motion]))
        data[i]['Y'][:, 0:3] = 0.0


        
# # jump sequences
#     for i in range(3,5):
#         data.append(GPy.util.datasets.cmu_mocap('16', ['0' + str(i+1-3)]))
#         data[i]['Y'][:, 0:3] = 0.0
#         index += data[i]['Y'].shape[0]
#         seq_index.append(index)
#     class_index.append(index)

# # boxing
#     for i in range(5,7):
#         data.append(GPy.util.datasets.cmu_mocap('14', ['0' + str(i+1-5)]))
#         data[i]['Y'][:, 0:3] = 0.0
#         index += data[i]['Y'].shape[0]
#         seq_index.append(index)
#     class_index.append(index)


    data = np.vstack([data[i]['Y'] for i in range(len(data))])

    return data
 

# #### classes
# 0, still,
# 1, talking on the phone,
# 2, writing on whiteboard,
# 3, drinking water,
# 4, rinsing mouth with water,
# 5, brushing teeth,
# 6, wearing contact lenses,
# 6, wearing contact lenses,
# 7, talking on couch,
# 8, relaxing on couch,
# 9, cooking (chopping),
# 10, cooking (stirring),
# 11, opening pill container,
# 11, opening pill container,
# 11, opening pill container,
# 12, working on computer,
# 13, random,



def test_adl():

    limit = 500

    result = []
    
    person = []
    for i in range(3):
        person.append(DatasetPerson(person=i+1))
        
        
    classes = {}
    class_counter = 0
    for personId in range(3):
        for activity, label in person[personId].activity_label.iteritems():
            if classes.get(label) is None:
                class_counter += 1
                classes[label] = {'number' : class_counter, 'samples':[]}

            person[personId].load_activity(activity)
            classes[label]['samples'].append(person[personId].get_processed_data())
        
        
    for label in classes.keys():
        cov = np.cov(np.vstack(tuple(classes[label]['samples'])).T)
        classes[label]['covariance'] = cov
        classes[label]['inverse_covariance'] = np.linalg.pinv(cov)

    testPerson = DatasetPerson(person=4)

    for testActivity, testLabel in testPerson.activity_label.iteritems():
        testPerson.load_activity(testActivity)
        testSample = testPerson.get_processed_data()

        for label, cl in classes.iteritems():
            for sample in cl['samples']:
                normalizer = min(len(sample), len(testSample))
                mahalanobis_distance = lambda x,y: scipy.spatial.distance.mahalanobis(x,y,cl['inverse_covariance'])

                mahalanobis_distance2 = lambda x,y: scipy.spatial.distance.mahalanobis(x,y,cl['covariance'])


                print testLabel + " --- " + label
                dtw_dist = dtw(testPerson.get_processed_data()[::10], sample[::10])/normalizer
                dtw_mah_dist = dtw(testPerson.get_processed_data()[::10], sample[::10], dist=mahalanobis_distance)/normalizer
                dtw_mah_dist2 = dtw(testPerson.get_processed_data()[::10], sample[::10], dist=mahalanobis_distance2)/normalizer

                

                print dtw_dist
                print dtw_mah_dist
                print dtw_mah_dist2

                result.append({'trueLabel' : testLabel, 'label': label,
                               'dtw': dtw_dist, 'dtw_mah': dtw_mah_dist,
                               'dtw_mah2': dtw_mah_dist2})

    return result
 

