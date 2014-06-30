import numpy as np
from GPy.core import Mapping
from utils import dtw


class SeqConstraints(Mapping):
    """
    Constraints for the GP-LVM optimization

    :param Y: observed data
    :type Y: np.ndarray

    """
    def __init__(self, Y, sequences, input_dim, output_dim=2):
        self.sequences = sequences
        self.seq_num = len(sequences)
        self.sequences.append(Y.shape[0])

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

    def g(self, Y_s, q):
        return sum([self.A[s, q] * dtw(Y_s, self.Y_s(s))
                    for s in range(self.seq_num)])

    def Y_s(self, s):
        return self.Y[self.sequences[s]:self.sequences[s+1]]

    def X_s(self, X, s):
        return X[self.sequences[s]:self.sequences[s+1]]

    def df_dtheta(self, dl_df, X):
        return np.hstack(([[self.g(self.Y_s(s), q) - np.mean(self.X_s(X, s))
                            for q in range(self.output_dim)]
                           for s in range(self.seq_num)]))

    def df_dX(self, dl_df, X):
        return dl_df

    def objective_function(self, x):
        self._set_params(x)
        return np.sum([[self.Lambda[s, q] * (self.g(self.Y_s(s), q)
                                             - np.mean(self.X_s(x, s)))
                        for q in range(self.output_dim)]
                       for s in range(self.seq_num)])

    def objective_function_gradients(self, x):
        self._set_params(x)
        df_dLambda = np.hstack(([[(self.g(self.Y_s(s), q)
                                   - np.mean(self.X_s(x, s)))
                                  for q in range(self.output_dim)]
                                 for s in range(self.seq_num)]))

        return np.hstack(([[0
                            for q in range(self.output_dim)]
                           for s in range(self.seq_num)]))


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
                 kernel=None, normalize_Y=False):

        self.seq_index = seq_index
        self.lagr_constraints = SeqConstraints(Y, seq_index, input_dim)
        GPLVM.__init__(self, Y, input_dim, init, X, kernel, normalize_Y)
        # use non-linear dim reduction method here
        # self.X =

    def objective_function(self, x):
        return super(SeqBCGPLVM, self).objective_function(x)

    def objective_function_gradients(self, x):
        return super(SeqBCGPLVM, self).objective_function_gradients(x)

    def objective_and_gradients(self, x):
        return super(SeqBCGPLVM, self).objective_and_gradients(x)


def test():
    import GPy as GPy
    data = []
    seq_index = [0]
    for i in range(3):
        data.append(GPy.util.datasets.cmu_mocap('35', ['0' + str(i+1)]))
        data[i]['Y'][:, 0:3] = 0.0
        seq_index.append(data[i]['Y'].shape[0])

    m = SeqBCGPLVM(np.vstack([data[i]['Y'] for i in range(3)]), 62, seq_index,
                   init='Random')

    return m
