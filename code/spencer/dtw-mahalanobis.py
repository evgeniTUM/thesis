import GPy
import numpy as np
import scipy.spatial

from utils import dtw
from dataset import DatasetPerson


sparsity = 1

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

    person = []
    for i in range(3):
        person.append(DatasetPerson(person=i+1))
        
        
    classes = {}
    class_counter = 0
    for personId in range(3):
        for activity, label in person[personId].activity_label.iteritems():
            if label == 'random':
                continue
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

    ground_truth = []
    prediction_mah = []
    prediction = []
    prediction_min = []
    prediction_mah_min = []

    for testActivity, testLabel in testPerson.activity_label.iteritems():
        if testLabel == 'random':
            continue

        testPerson.load_activity(testActivity)
        testSample = testPerson.get_processed_data()
        
        ground_truth.append(classes[testLabel]['number'])

        score_mah = []
        score = []
        score_min = []
        score_mah_min = []
        
        for label, cl in classes.iteritems():
            
            sample_prediction = []
            sample_prediction_mah = []

            for sample in cl['samples']:
                normalizer = min(len(sample), len(testSample))

                mahalanobis_distance = lambda x,y: scipy.spatial.distance.mahalanobis(x,y,cl['inverse_covariance'])


                dtw_dist = dtw(testPerson.get_processed_data()[::sparsity], sample[::sparsity])/normalizer
                dtw_mah_dist = dtw(testPerson.get_processed_data()[::sparsity], sample[::sparsity], dist=mahalanobis_distance)/normalizer



                print testLabel, '---', label
                print dtw_dist, dtw_mah_dist

                sample_prediction_mah.append(dtw_mah_dist)
                sample_prediction.append(dtw_dist)
                
            score_mah.append(sum(sample_prediction_mah)/len(cl['samples']))
            score.append(sum(sample_prediction)/len(cl['samples']))
            score_mah_min.append(min(sample_prediction_mah))
            score_min.append(min(sample_prediction))
            

        prediction_mah.append(np.argmin(score_mah))
        prediction.append(np.argmin(score))
        prediction_mah_min.append(np.argmin(score_mah_min))
        prediction_min.append(np.argmin(score_min))



    m = []
    for label, cl in classes.iteritems():
        m.append(cl['number'])

    prediction_mah = map(lambda x: m[x], prediction_mah)
    prediction = map(lambda x: m[x], prediction)
    prediction_mah_min = map(lambda x: m[x], prediction_mah_min)
    prediction_min = map(lambda x: m[x], prediction_min)


    return ground_truth, prediction_mah, prediction, prediction_mah_min, prediction_min
 

