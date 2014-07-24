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
        return np.sum([[self.Lambda[s, q] * (self.g(s, q)
                                             - self.mu(s, q))
                        for q in range(self.output_dim)]
                       for s in range(self.seq_num)])

    def objective_function_gradients(self, x):
        self._set_params(x)
        return np.hstack((self.df_dLambda(), self.df_dA())).flatten()

    def objective_and_gradients(self, x):
        return self.objective_function(x), self.objective_function_gradients(x)


from GPy.models import GPLVM


class SeqBCGPLVM(GPLVM):
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
                 kernel=None, normalize_Y=False, discriminative_sigma=0.5):

        self.seq_index = seq_index
        self.lagr_constraints = SeqConstraints(self, Y, seq_index, input_dim)
        GPLVM.__init__(self, Y, input_dim, init, X, kernel, normalize_Y)

        self.prior = DiscriminativePrior(discriminative_sigma, seq_index)
        # use non-linear dim reduction method here
        # self.X =

    def objective_function(self, x):
        return super(SeqBCGPLVM, self).objective_function(x[:self.num_params_transformed()]) + \
            self.lagr_constraints.objective_function(x[self.num_params_transformed():])

    def objective_function_gradients(self, x):
        return np.hstack((super(SeqBCGPLVM, self).objective_function_gradients(x[:self.num_params_transformed()]),
                          self.lagr_constraints.objective_function_gradients(x[self.num_params_transformed():])))

    def objective_and_gradients(self, x):
        return np.hstack((super(SeqBCGPLVM, self).objective_and_gradients(x[:self.num_params_transformed()]),
                          self.lagr_constraints.objective_function_gradients(x[self.num_params_transformed()])))

    def log_prior(self):
        return self.prior.lnpdf(self._get_params())

    def _log_prior_gradients(self):
        return self.prior.lnpdf_grad(self._get_params())

    def optimize(self, optimizer=None, start=None, **kwargs):
        if start is None:
            start = np.hstack((self._get_params_transformed(),
                               self.lagr_constraints._get_params()))

        super(SeqBCGPLVM, self).optimize(optimizer, start, **kwargs)

    def log_prior(self):
        return self.prior.lnpdf(self.X)

    def _log_prior_gradients(self):
        return self.prior.lnpdf_grad(self.X)


_data_ = None


def createModel(sigma=0.5):
    import GPy as GPy
    global _data_

    data = []
    seq_index = [0]
    length = 0
    for i in range(3):
        data.append(GPy.util.datasets.cmu_mocap('35', ['0' + str(i+1)]))
        data[i]['Y'][:, 0:3] = 0.0
        length += data[i]['Y'].shape[0]
        seq_index.append(length)

    m = SeqBCGPLVM(np.vstack([data[i]['Y'] for i in range(3)]),
                   2, seq_index, init='PCA', discriminative_sigma=sigma)

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
