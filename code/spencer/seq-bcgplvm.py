import numpy as np
import math
from GPy.core import Mapping
from utils import dtw
from priors import DiscriminativePrior


class SeqConstraints(Mapping):
    """
    Constraints for the GP-LVM optimization

    :param Y: observed data
    :type Y: np.ndarray

    """
    def __init__(self, model, Y, sequences, input_dim, output_dim=2):
        self.model = model
        self.sequences = sequences
        self.seq_num = len(sequences) - 1

        self.Y = Y
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Lambda = np.array((self.seq_num, self.output_dim))
        self.A = np.array((self.seq_num, self.output_dim))
        self.num_params = 2*(self.seq_num*self.output_dim)

        self.K = np.array([[dtw(self.Y_s(x), self.Y_s(y))
                            for x in range(self.seq_num)]
                           for y in range(self.seq_num)])
        self.randomize()

    def _get_params(self):
        return np.hstack((self.Lambda.flatten(), self.A.flatten()))

    def _set_params(self, x):
        self.Lambda = x[:self.seq_num*self.output_dim] \
            .reshape(self.seq_num, self.output_dim).copy()

        self.A = x[self.seq_num*self.output_dim:] \
            .reshape(self.seq_num, self.output_dim).copy()

    def randomize(self):
        self.Lambda = np.random.randn(self.seq_num, self.output_dim) \
            / np.sqrt(self.seq_num+1)

        self.A = np.random.randn(self.seq_num, self.output_dim) \
            / np.sqrt(self.seq_num+1)

    def g(self, current_sequence, q):
        return sum([self.A[s, q] * math.exp(-self.K[current_sequence, s])
                    for s in range(self.seq_num)])

    def Y_s(self, s):
        return self.Y[self.sequences[s]:self.sequences[s+1]]

    def mu(self, s, q):
        X = self.model.X
        return np.mean(X[self.sequences[s]:self.sequences[s+1], q])

    def df_dLambda(self):
        return [[self.g(s, q)
                 - self.mu(s, q)
                 for q in range(self.output_dim)]
                for s in range(self.seq_num)]

    def df_dA(self):
        return [[self.Lambda[s, q] * sum([math.exp(-self.K[s, j])
                                         for j in range(self.seq_num)])
                 for q in range(self.output_dim)]
                for s in range(self.seq_num)]

    def df_dX(self):
        return None

    def objective_function(self, x):
        self._set_params(x)
        return -np.sum([[self.Lambda[s, q] * (self.g(s, q)
                                              - self.mu(s, q))
                         for q in range(self.output_dim)]
                        for s in range(self.seq_num)])

    def objective_function_gradients(self, x):
        self._set_params(x)
        return -np.hstack((self.df_dLambda(), self.df_dA())).flatten()

    def objective_and_gradients(self, x):
        return self.objective_function(x), self.objective_function_gradients(x)


from GPy.models import GPLVM
from GPy.models import SparseGPLVM
from GPy.models import BCGPLVM


class SeqBCGPLVM(SparseGPLVM):
    """
    Sequence back-constrained Gaussian Process Latent Variable Model

    See paper:
    Valsamis Ntouskos, Panagiotis Papadakis, Fiora Pirri:
    Discriminative Sequence Back-constrained GP-LVM for MOCAP based
    Action  Recognition


    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, seq_index, init='PCA', X=None,
                 kernel=None, normalize_Y=False, sigma=0.5):

        self.sigma = float(sigma)
        self.seq_index = seq_index
        self.lagr_constraints = SeqConstraints(self, Y, seq_index, input_dim)


        SparseGPLVM.__init__(self, Y, input_dim, init=init, kernel=kernel, num_inducing=100)


        self.prior = DiscriminativePrior(seq_index)

    def objective_function(self, x):
        return super(SeqBCGPLVM, self).objective_function(x[:self.num_params_transformed()]) + \
            self.lagr_constraints.objective_function(x[self.num_params_transformed():])

    def objective_function_gradients(self, x):
        return np.hstack((super(SeqBCGPLVM, self).objective_function_gradients(x[:self.num_params_transformed()]),
                          self.lagr_constraints.objective_function_gradients(x[self.num_params_transformed():])))

    def objective_and_gradients(self, x):
        return self.objective_function(), self.objective_function_gradients()


    def optimize(self, optimizer=None, start=None, **kwargs):
        if start is None:
            start = np.hstack((self._get_params_transformed(),
                               self.lagr_constraints._get_params()))

        super(SeqBCGPLVM, self).optimize(optimizer, start, **kwargs)

    def log_prior(self):
        return (1.0/self.sigma**2) * self.prior.lnpdf(self.X) 
        
    
    def _log_prior_gradients(self):
        return (1.0/self.sigma**2) * np.hstack((self.prior.lnpdf_grad(self.X), 
                                                np.zeros(self._get_params().size - self.X.size))) 
            


_data_ = None


def createModel(sigma=0.5, init='PCA', lengthscale=1.0):
    import GPy as GPy
    global _data_

    data = []
    seq_index = [0]
    index = 0
# walk sequences
    for i in range(2):
        data.append(GPy.util.datasets.cmu_mocap('35', ['0' + str(i+1)]))
        data[i]['Y'][:, 0:3] = 0.0
        index += data[i]['Y'].shape[0]
    seq_index.append(index)

    # jump sequences
    for i in range(2,4):
        data.append(GPy.util.datasets.cmu_mocap('16', ['0' + str(i+1-2)]))
        data[i]['Y'][:, 0:3] = 0.0
        index += data[i]['Y'].shape[0]
    seq_index.append(index)

# # boxing
#     for i in range(5,7):
#         data.append(GPy.util.datasets.cmu_mocap('14', ['0' + str(i+1-5)]))
#         data[i]['Y'][:, 0:3] = 0.0
#         index += data[i]['Y'].shape[0]
#         seq_index.append(index)

    m = SeqBCGPLVM(np.vstack([data[i]['Y'] for i in range(len(data))]),
                   2, seq_index, sigma=sigma, init=init)

    _data_ = data
    return m


def createSeqModel(sigma=0.5, lengthscale=1.0):
    """ used to test sequence back constraints """
    import GPy as GPy
    global _data_

    seq_index = [0]


    data = []
    data.append(np.array([ [1, 2, 3],
                           [2, 3, 4],
                           [3, 4, 5]]))
    seq_index.append(3)

    data.append(np.array([ [1, 2, 3],
                           [1, 2, 4],
                           [1, 2, 5]]))
    seq_index.append(6)

    data.append(np.array( [ [1, 2, 3],
                            [2, 2, 3],
                            [3, 2, 3]]))
    seq_index.append(9)
    data.append(np.array( [ [5, 6, 7],
                            [4, 5, 6],
                            [3, 3, 3]]))

    seq_index.append(12)




    actions = len(data)


    m = SeqBCGPLVM(np.vstack([data[i] for i in range(actions)]),
                   2, seq_index, sigma=sigma)

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
