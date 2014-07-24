import numpy as np
from GPy.core.domains import REAL


class DiscriminativePrior:
    """
    Prior to implement the 'Discriminative GP-LVM'
    See: - Urtasun, Darrell (2007). Discriminative Gaussian process
    latent variable model for classification.
    http://dl.acm.org/citation.cfm?id=1273613
    """
    domain = REAL

    def __init__(self, sigma, seq_index):
        self.sigma = sigma
        self.seq_index = seq_index
        return

    def _J(self, X):
        return np.trace(np.dot(np.linalg.inv(self._S_b(X)), self._S_w(X)))

    def _S_w(self, X):
        dim = X.shape[1]
        N = float(X.shape[0])

        result = np.zeros((dim, dim))
        M_0 = np.mean(X, axis=0)
        for i in range(len(self.seq_index)-1):
            M_i = np.mean(self._X_i(X, i), axis=0)
            result += self._N_i(i)/N * np.outer(M_i - M_0, M_i - M_0)

        return result

    def _S_b(self, X):
        dim = X.shape[1]
        N = float(X.shape[0])

        result = np.zeros((dim, dim))
        for i in range(len(self.seq_index)-1):
            M_i = np.mean(self._X_i(X, i), axis=0)
            result += self._N_i(i)/N * np.sum(
                [np.outer((x_k - M_i), (x_k - M_i))
                 for x_k in self._X_i(X, i)],
                axis=0)

        return result

    def _X_i(self, X, i):
        return X[self.seq_index[i]:self.seq_index[i+1]]

    def _N_i(self, i):
        return self.seq_index[i+1] - self.seq_index[i]

    def summary(self):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.lnpdf(x))

    def lnpdf(self, x):
        return  (1/self.sigma**2) * self._J(x)

    def lnpdf_grad(self, x):
        return self._J(x)/(self.sigma**2)
