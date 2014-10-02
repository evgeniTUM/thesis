## This is an implementation of:
# - Zhang, Tian (2012). RGB-D Camera-based Daily Living Activity Recognition.
#  Journal of Computer Vision and Image Processing.
#  http://www-ee.ccny.cuny.edu/www/web/yltian/Publications/NWPJ-201209-15.pdf


from dataset import DatasetPerson
from scipy.cluster.vq import *
import sklearn.svm
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import sklearn.metrics



classes = None


def create_code_book(features, size=100):
    # whitened_features = whiten(features)
    
    indices = {}
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=size, n_init=10)
    kmeans.fit(features)
    
    # code_book = kmeans2(whitened_features, size, minit='points')
    
    return kmeans

def learn(persons=[1,2,3], clusters=128):

    samples, labels = read_persons(persons)

    data = np.vstack(samples)

    print "creating code book with", clusters, "clusters and", data.shape[0], "features"
    kmeans = create_code_book(data, size=clusters)


    lin_svm = sklearn.svm.LinearSVC()
    
    # get nearest centroids
    codes = []
    for sample in samples:
        codes.append(kmeans.predict(sample))

    # histogram pooling
    histograms = []
    for code in codes:
        print code
        histogram = compute_histogram(code, clusters)
        histograms.append(histogram)
        
    lin_svm.fit(histograms, labels)
    
    return lin_svm, kmeans

def predict(lin_svm, kmeans, sample):
    return lin_svm.predict(compute_histogram(kmeans.predict(sample), kmeans.n_clusters))

        
def compute_histogram(code, clusters):
    histogram = [0]*clusters
    for centroid in code:
        histogram[centroid] += 1

    histogram = np.array(histogram) / float(len(code))
    return histogram

def read_persons(persons = [1,2,3]):
    global classes

    print "Learning persons", persons

    person = []
    for i in persons:
        person.append(DatasetPerson(person=i))

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
            classes[label]['samples'].append(person[personId].get_features())
            
            sample_features = person[personId].get_features()

            
            samples.append(sample_features)
            labels.append(classes[label]['number'])


    return samples, labels


def test(person=4, test_partial=False):
    persons=[1,2,3,4]
    persons.remove(person)

    lin_svm, kmeans = learn(persons) 

    test_person(lin_svm, kmeans, person, test_partial)



def test_person(lin_svm, kmeans, person=4, test_partial=False):
    global classes
    
    testPerson = DatasetPerson(person=person)

    ground_truth = []
    prediction = []

    for testActivity, testLabel in testPerson.activity_label.iteritems():
        if testLabel == 'random':
            continue

        testPerson.load_activity(testActivity)
        testSample = testPerson.get_features()
        
        ground_truth.append(classes[testLabel]['number'])

        if test_partial:
            frames = 50
            start_frame = np.random.randint(0, testSample.shape[0]-frames)
            testSample = testSample[start_frame:start_frame+frames]


        prediction.append(predict(lin_svm, kmeans, testSample)[0])
        
        


    return ground_truth, prediction
