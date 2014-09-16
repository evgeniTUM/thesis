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

    def __init__(self, gamma, seq_index):
        self.gamma = float(gamma)
        self.seq_index = seq_index
        return

    def setup(self, X):
        self.dim = X.shape[1]
        self.N = X.shape[0]

        self.mean_X = np.mean(X, axis=0)
        self.S_w = np.zeros((self.dim, self.dim))
        self.S_b = np.zeros((self.dim, self.dim))
        self.W = np.zeros((self.N, self.N))
        self.B = np.zeros((self.N, self.N))
        self.num_classes = len(self.seq_index)-1
        G = np.zeros((self.N, self.num_classes))
        N_s = []

        for i in range(self.num_classes):
            index = slice(self.seq_index[i],self.seq_index[i+1])
            
            N_i = self.seq_index[i+1] - self.seq_index[i]
            M_i = np.mean(X[index], axis=0)
            
            self.W[index,index] = np.eye(N_i) - (1.0/N_i) * np.ones((N_i, N_i))
            G[index,i] = (1.0/N_i) * np.ones((N_i))
            N_s.append(N_i)
            
        B_p = np.diag(N_s)
        G = G - (1.0/self.N)*np.ones((self.N, self.num_classes))
        self.B = np.dot(np.dot(G, B_p), G.T)

        self.S_w = (1.0/self.N)* np.dot(np.dot(X.T, self.W), X)
        self.S_b = (1.0/self.N)* np.dot(np.dot(X.T, self.B), X)

        self.S_b_inv = np.linalg.inv(self.S_b)
        self.A = np.dot(self.S_b_inv, self.S_w)
        
    def _dJ_dX(self, X):
        self.setup(X)
        
        result = np.zeros((self.N, self.dim))
        for d in range(self.dim):
            for c in range(self.num_classes):
                for i in range(self.seq_index[c], self.seq_index[c+1]):
                    N_i = self.seq_index[c+1] - self.seq_index[c]
                    dx_dX = np.zeros(X.T.shape)
                    dx_dX[d, i] = 1.0
                    
                    dSw_dX = np.dot(np.dot(dx_dX, self.W), dx_dX.T)
                    dSb_dX = np.dot(np.dot(dx_dX, self.B), dx_dX.T)

                    temp = -np.dot(dSb_dX, self.A) + dSw_dX
                    result[i, d] = result[i, d] + np.trace(
                        (2.0/self.N) * np.dot(self.S_b_inv, temp)) 
        return result

    def _J(self, X):
        self.setup(X)
        return np.trace(self.A)

    def summary(self):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.lnpdf(x))

    def lnpdf(self, x):
        return self.gamma * self._J(x)

    def lnpdf_grad(self, x):
        return self.gamma * self._dJ_dX(x).flatten()
