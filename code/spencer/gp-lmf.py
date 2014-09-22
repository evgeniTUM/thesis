import numpy as np
import math


from GPy.models import GPLVM
from GPy.models import SparseGPLVM
from GPy.models import BCGPLVM
from GPy.models import GPRegression
import GPy

class GPLMF(BCGPLVM):
    """
    Gaussian Process - Latent Motion Flow
    Activity modeling by sparse motion flow model in latent space.


    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, seq_index, init='PCA', X=None,
                 kernel=None, normalize_Y=False, sigma=0.5, mapping=None, class_index=None):

        self.sigma = float(sigma)
        self.seq_index = seq_index
        self.class_index = class_index
        self.labels = seq_index2labels(class_index)

        if mapping == None:
            GPLVM.__init__(self, Y, input_dim, init, X, kernel, normalize_Y)
            #SparseGPLVM.__init__(self, Y, input_dim, kernel=kernel, init=init, num_inducing=20)
        else:
            print "Using: back-constraints"
            BCGPLVM.__init__(self, Y, input_dim, kernel=kernel, mapping=mapping)



def calc_y(X):
    dimensions = X.shape[1]
    result = []
    for d in range(dimensions):
        df_dx = [ [X[i+1,d] - X[i-1,d]] for i in range(1,X.shape[0]-1) ]
        result.append(df_dx)

    return X[1:-1], np.array(result)


def learn_flow(X, y, l_scale=0.1, variance=1.0):

    dimensions = X.shape[1]
    lengthscales = [ l_scale for d in range(dimensions)]
    kernel = GPy.kern.rbf(dimensions, ARD=True, 
                          lengthscale=lengthscales, variance=variance)
    m = GPRegression(X,y,kernel)

    m.optimize('bfgs', max_iters=200)
    
    return m

def learn_flows(X, Y, l_scale=1.0, variance=0.00001 ):
    return [ learn_flow(X, Y[d], l_scale, variance)
             for d in range(Y.shape[0])]
        


def energy(X, flow):
    return 0



def plot_flow_field(f, model, index=0):
    import matplotlib 
    import matplotlib.pyplot as plt

    limit = max(abs(np.min(model.X)), np.max(model.X))
    dimensions = len(f)
    samples = 1000**(1.0/dimensions)

    
    x = []
    for d in range(dimensions):
        x.append(np.linspace(-limit, limit, samples))

    x = list(np.meshgrid(*x))
    X = np.array(zip(*[np.array(el).flatten() for el in x]))

    vx = []
    for d in range(dimensions):
        vx.append(f[d].predict(X)[index])
    

    # plt.streamplot(x, y, vx, vy, color=)


    if dimensions == 2:
        plot(model);
        quiver(x[0], x[1], vx[0], vx[1])
    else:
        if dimensions == 3:
            from mpl_toolkits.mplot3d import axes3d

            fig = plot3d(model)
            ax = fig.gca(projection='3d')

            ax.quiver3D(x[0].reshape(vx[0].shape[0],1), 
                        x[1].reshape(vx[0].shape[0],1),
                        x[2].reshape(vx[0].shape[0],1), 
                        vx[0], vx[1], vx[2],
                        length=0.1)

    plt.show()
    



def test(m):
    x, y = calc_y(m.X)
    f = learn_flows(x, y)

    plot_flow_field(f, m)

    
_data_ = None



def get_mocap_data():
    data = []
    seq_index = [0]
    class_index = [0]
    index = 0

    # walk sequences
    for i in range(1):
        data.append(GPy.util.datasets.cmu_mocap('35', ['0' + str(i+1)]))
        data[i]['Y'][:, 0:3] = 0.0
        index += data[i]['Y'].shape[0]
        seq_index.append(index)
    class_index.append(index)


        
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

    return data, seq_index, class_index



def createModel(sigma=0.5, init='Random', lengthscale=1.0, dimensions=2):
    global _data_

    data, seq_index, class_index = get_mocap_data()

    back_kernel=GPy.kern.rbf(data.shape[1], lengthscale=lengthscale)
    mapping = GPy.mappings.Kernel(X=data, output_dim=dimensions, kernel=back_kernel)

    m = GPLMF(data, dimensions, seq_index, class_index=class_index,
              sigma=sigma, init=init, mapping=mapping)

    _data_ = data
    return m


def seq_index2labels(seq_index):
    labels = np.ones((seq_index[-1]))
    for i in range(len(seq_index)-1):
        labels[seq_index[i]:seq_index[i+1]] = i

    return labels




def plot(m, visual=False):
    import GPy
    global _data_

    ax = m.plot_latent(seq_index2labels(m.seq_index))

    if visual:
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.skeleton_show(y[None, :], _data_[0]['skel'])
        lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')
        lvm_visualizer.close()


def plot3d(m):
    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(m.X[:,0], m.X[:,1], m.X[:,2])

    plt.show()
    return fig
