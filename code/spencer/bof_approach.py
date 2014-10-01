from dataset import DatasetPerson
from scipy.cluster.vq import *
from sklearn import svm
from sklearn.cluster import KMeans

classes = None


def create_code_book(features, size=100):
    # whitened_features = whiten(features)
    
    indices = {}
    kmeans = KMeans(init='k-means++', n_clusters=size, n_init=10)
    kmeans.fit(features)
    # code_book = kmeans2(whitened_features, size, minit='points')
    
    return kmeans

def learn_svm(clusters=128):
    samples, labels = learn_persons()

    data = np.vstack(samples)

    print "creating code book with", clusters, "clusters and", data.shape[0], "features"
    kmeans = create_code_book(data, size=clusters)


    lin_svm = svm.LinearSVC()
    
    # get nearest centroids
    codes = []
    for sample in samples:
        codes.append(kmeans.predict(sample))

    # histogram pooling
    histograms = []
    for code in codes:
        histogram = compute_histogram(code, clusters)
        histograms.append(histogram)
        
    lin_svm.fit(histograms, labels)
    
    return lin_svm, kmeans, histograms, labels

def predict(lin_svm, kmeans, sample):
    return lin_svm.predict(compute_histogram(kmeans.predict(sample), kmeans.n_clusters))

        
def compute_histogram(code, clusters):
    histogram = [0]*clusters
    for centroid in code:
        histogram[centroid] += 1

    return histogram

def learn_persons(persons = [1,2,3]):
    global classes

    person = []
    for i in persons:
        person.append(DatasetPerson(person=i+1))

    samples = []
    labels = []

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
            
            sample_features = person[personId].get_features()

            
            samples.append(sample_features)
            labels.append(classes[label]['number'])


    return samples, labels


def test_person(person=4):
    global classes
    lin_svm, kmeans, histograms, labels = learn_svm() 
    

    testPerson = DatasetPerson(person=person)

    ground_truth = []
    prediction = []

    for testActivity, testLabel in testPerson.activity_label.iteritems():
        if testLabel == 'random':
            continue

        testPerson.load_activity(testActivity)
        testSample = testPerson.get_features()
        
        ground_truth.append(classes[testLabel]['number'])
        prediction.append(predict(lin_svm, kmeans, testSample)[0])
        
        


    return ground_truth, prediction
