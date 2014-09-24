import GPy
import numpy as np

from utils import dtw

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
 


def test(data, x, y):
    cov = np.cov(data.T)

    cov_inv = np.linalg.pinv(cov)
    dist = lambda x,y: scipy.spatial.distance.mahalanobis(x,y,cov_inv)
    return dtw(x,y,dist=dist)
