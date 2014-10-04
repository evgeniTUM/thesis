## This is an implementation of:
# - Zhang, Tian (2012). RGB-D Camera-based Daily Living Activity Recognition.
#  Journal of Computer Vision and Image Processing.
#  http://www-ee.ccny.cuny.edu/www/web/yltian/Publications/NWPJ-201209-15.pdf

from dataset import DatasetPerson
from dataset import read_persons
from utils import lcs,dtw,euclidean_dist

import sklearn.svm
import sklearn.cluster
import sklearn.metrics
import sklearn.neighbors
import numpy as np
import pickle


def create_code_book(features, size=100):
    print "creating code book with", size, "clusters from", features.shape[0], "features"
 
    kmeans = sklearn.cluster.MiniBatchKMeans(init='k-means++', n_clusters=size, n_init=10)
    kmeans.fit(features)
        
    return kmeans


def learn_ext_lcs(samples, labels, clusters=10):

    centroids = []
    for i in range(max(labels)+1):
        data = np.vstack([sample
                          for sample, label in zip(samples, labels)
                          if label == i])
        kmeans = create_code_book(data, size=clusters)
        centroids.append(kmeans.cluster_centers_)

    centroids = np.vstack(centroids)

    nn_tree = sklearn.neighbors.BallTree(centroids)
    nn_classificator = NN_Classificator(max(labels)+1)

    
    # get nearest centroids
    codes = []
    for sample in samples:
        code = (nn_tree.query(sample, k=1, return_distance=False))
        codes.append([el[0] for el in code])

    nn_classificator.fit(codes, labels)

    
    
    return nn_classificator, nn_tree



def learn_ext(samples, labels, clusters=10):

    centroids = []
    for i in range(max(labels)+1):
        data = np.vstack([sample
                          for sample, label in zip(samples, labels)
                          if label == i])
        kmeans = create_code_book(data, size=clusters)
        centroids.append(kmeans.cluster_centers_)

    centroids = np.vstack(centroids)

    nn_tree = sklearn.neighbors.BallTree(centroids)

    classificator = sklearn.svm.LinearSVC()
    
    # get nearest centroids
    codes = []
    for sample in samples:
        codes.append(nn_tree.query(sample, k=1, return_distance=False))


    # histogram pooling
    histograms = []
    for code in codes:
        histogram = compute_histogram(code, centroids.shape[0])
        histograms.append(histogram)
     
    
    classificator.fit(histograms, labels)
    
    return classificator, nn_tree


def learn(samples, labels, clusters=128):
    data = np.vstack(samples)

    kmeans = create_code_book(data, size=clusters)

    classificator = sklearn.svm.LinearSVC()
    #classificator = GaussianProcess(theta0=5e-1)
    
    # get nearest centroids
    codes = []
    for sample in samples:
        codes.append(kmeans.predict(sample))

    # histogram pooling
    histograms = []
    for code in codes:
        histogram = compute_histogram(code, clusters)
        histograms.append(histogram)
     
    
    classificator.fit(histograms, labels)
    
    return classificator, kmeans


        
def compute_histogram(code, clusters):

    histogram = [0]*clusters
    for centroid in code:
        histogram[centroid] += 1

    histogram = np.array(histogram) / float(len(code))
    return histogram




def test(person=4, test_partial=False):
    persons=[1,2,3,4]
    persons.remove(person)
    samples, labels = read_persons(persons)
    testSamples, testLabels = read_persons(persons=[person])
    
    classificator, kmeans = learn(samples, labels) 

    test_person(classificator, kmeans, testSamples, testLabels, test_partial)



def test_data(classificator, nn_tree, testSamples, testLabels, test_partial=False):

    ground_truth = []
    prediction = []

    for testSample, testLabel in zip(testSamples, testLabels):
      
        ground_truth.append(testLabel)

        if test_partial:
            frames = 50
            start_frame = np.random.randint(0, testSample.shape[0]-frames)
            testSample = testSample[start_frame:start_frame+frames]


        prediction.append(predict(classificator, nn_tree, testSample)[0])
        #prediction.append(predict_lcs(classificator, nn_tree, testSample))
        #prediction.append(predict_dtw(classificator, nn_tree, testSample))

    return ground_truth, prediction







class NN_Classificator():
    def __init__(self, classes):
        self.n_classes = classes
        self.classes = []  
        for i in range(classes):
            self.classes.append([])

    def fit(self, samples, labels):
        for i in range(len(labels)):
            self.classes[labels[i]].append(samples[i])



def predict(classificator, nn_tree, sample):
    print nn_tree.predict(sample)

    if isinstance(nn_tree, sklearn.neighbors.BallTree):
        return classificator.predict(compute_histogram(
            nn_tree.query(sample, k=1, return_distance=False),
            nn_tree.get_arrays()[0].shape[0]))
    else:
        return classificator.predict(compute_histogram(nn_tree.predict(sample), nn_tree.n_clusters))




def predict_lcs(lcs_classificator, nn_tree, testSample, use_min=False):
    testSample = nn_tree.query(testSample, k=1, return_distance=False)
    testSample = [sample[0] for sample in testSample ]

    scores = []
    for cl in lcs_classificator.classes:
        class_score = []
        for sample in cl:

            intersection = set(testSample).intersection(sample)
            test = filter(lambda x: x in intersection, testSample)
            sample = filter(lambda x: x in intersection, sample)
            
            if len(intersection) > 0:
                class_score.append(lcs(test, sample))
            else:
                class_score.append(0.0)

        print scores
        scores.append(sum(np.array(class_score)) / float(len(cl)))
            
    return np.argmax(scores)



def predict_dtw(lcs_classificator, nn_tree, testSample, use_min=False):

    testSample = nn_tree.query(testSample, k=1, return_distance=False)
    testSample = [sample[0] for sample in testSample ]

    n_centroids = nn_tree.get_arrays()[0].shape[0]

    distances = np.ndarray((n_centroids, n_centroids))
    for i in range(n_centroids):
        for j in range(n_centroids):
            distances[i,j] = euclidean_dist(nn_tree.get_arrays()[0][i],
                                            nn_tree.get_arrays()[0][j])


    dist = lambda x,y: distances[x,y]


    scores = []
    for cl in lcs_classificator.classes:
        class_score = []
        for sample in cl:
            class_score.append(dtw(testSample[::10], sample[::10], dist=dist)/min(len(sample), len(testSample)))

        print scores
        scores.append(sum(np.array(class_score)) / float(len(cl)))
            
    return np.argmin(scores)
