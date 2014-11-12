import numpy as np
import pylab as pb
import sys, pdb
from GPy.core import GP
from GPy.models import SparseGPLVM
from GPy.mappings import Kernel

class SparseBCGPLVM(SparseGPLVM):
    """
    Back constrained Gaussian Process Latent Variable Model
    with sparse GPs

    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'
    :param mapping: mapping for back constraint
    :type mapping: GPy.core.Mapping object

    """
    def __init__(self, Y, input_dim, init='PCA', X=None, kernel=None,  normalize_Y=False, mapping=None):
        
        if mapping is None:
            mapping = Kernel(X=Y, output_dim=input_dim)
        self.mapping = mapping
        SparseGPLVM.__init__(self, Y, input_dim, kernel=kernel, init=init)
        self.X = self.mapping.f(self.likelihood.Y)

    def _get_param_names(self):
        return self.mapping._get_param_names() + GP._get_param_names(self)

    def _get_params(self):
        return np.hstack((self.mapping._get_params(), GP._get_params(self)))

    def _set_params(self, x):
        self.mapping._set_params(x[:self.mapping.num_params])
        self.X = self.mapping.f(self.likelihood.Y)
        GP._set_params(self, x[self.mapping.num_params:])

    def _log_likelihood_gradients(self):
        dL_df = self.kern.dK_dX(self.dL_dK, self.X)
        dL_dtheta = self.mapping.df_dtheta(dL_df, self.likelihood.Y)
        return np.hstack((dL_dtheta.flatten(), GP._log_likelihood_gradients(self)))


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

    back_kernel=GPy.kern.rbf(data[0]['Y'].shape[1], lengthscale=lengthscale)
    mapping = GPy.mappings.Kernel(X=data[0]['Y'], output_dim=2, kernel=back_kernel)
    m = SparseBCGPLVM(np.vstack([data[i]['Y'] for i in range(len(data))]),
                      2, seq_index, init=init, mapping=mapping, kernel=GPy.kern.rbf(data[0]['Y'].shape[0] ))

    _data_ = data
    return m
