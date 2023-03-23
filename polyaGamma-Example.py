# https://gregorygundersen.com/blog/2019/09/20/polya-gamma/
import matplotlib.pyplot as plt
import numpy as np
from   numpy.linalg import inv
import numpy.random as npr
from   pypolyagamma import PyPolyaGamma

def sigmoid(x):
    """Numerically stable sigmoid function.
    """
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def multi_pgdraw(pg, B, C):
    """Utility function for calling `pgdraw` on every pair in vectors B, C.
    """
    arr = np.array([pg.pgdraw(b, c).set_trunc(3) for b, c in zip(B, C)])
    #arr = np.array([pg.pgdraw(b, c) for b, c in zip(B, C)])
    return arr

def gen_bimodal_data(N, p):
    """Generate bimodal data for easy sanity checking.
    """
    y     = npr.random(N) < p
    X     = np.empty(N)
    X[y]  = npr.normal(3, 1, size=y.sum())
    X[~y] = npr.normal(4, 1.4, size=(~y).sum())
    return X.astype(int), y.astype(int)


# Set priors and create data.
N_train = 1000
N_test  = 1000
b       = np.zeros(2)
B       = np.diag(np.ones(2))
X_train, y_train = gen_bimodal_data(N_train, p=0.3)
X_test, y_test   = gen_bimodal_data(N_test, p=0.3)
# Prepend 1 for the bias β_0.
X_train = np.vstack([np.ones(N_train), X_train])
X_test  = np.vstack([np.ones(N_test), X_test])

# Peform Gibb sampling for T iterations.
pg         = PyPolyaGamma()
T          = 100
Omega_diag = np.ones(N_train)
beta_hat   = npr.multivariate_normal(b, B)
k          = y_train - 1/2.

for _ in range(T):
    # ω ~ PG(1, x*β).
    Omega_diag = multi_pgdraw(pg, np.ones(N_train), X_train.T @ beta_hat.astype(int))
    # β ~ N(m, V).
    V         = inv(X_train @ np.diag(Omega_diag) @ X_train.T + inv(B))
    m         = np.dot(V, X_train @ k + inv(B) @ b)
    beta_hat  = npr.multivariate_normal(m, V)

PyPolyaGamma().pgdrawv(321,54, 65)
#
# y_pred = npr.binomial(1, sigmoid(X_test.T @ beta_hat.astype(int)))
# bins = np.linspace(X_test.min()-3., X_test.max()+3, 100)
# plt.hist(X_test.T[y_pred == 0][:, 1],    color='r', bins=bins)
# plt.hist(X_test.T[~(y_pred == 0)][:, 1], color='b', bins=bins)
# plt.show()
#
#
# import pandas as pd
# from patsy import dmatrices, dmatrix
# import numpy as np
# dfPexorig_wide = pd.read_csv("dfPexorig_wide.csv")
# counts = np.array(dfPexorig_wide.iloc[:,3:])
#
#
# X = np.ones((6,2))
# X[:3,1] = 0
# X = np.transpose(X)
# counts = np.array(dfPexorig_wide.iloc[:,3:])
#
#
# Omega_diag = multi_pgdraw(pg, np.ones(1000), np.transpose(X) @ beta_hat)
# V  = inv(counts @ np.diag(Omega_diag) @  np.transpose(counts))

'''
https://github.com/slinderman/pypolyagamma
'''

from pypolyagamma import logistic, PyPolyaGamma
import numpy as np
from numpy import random as npr

# Consider a simple binomial model with unknown probability
# Model the probability as the logistic of a scalar Gaussian.
N = 10
mu = 0.0
sigmasq = 1.0
x_true = npr.normal(mu, np.sqrt(sigmasq))
p_true = logistic(x_true)
y = npr.binomial(N, p_true)

N_samples = 10000
pg = PyPolyaGamma(seed=0)
xs = np.zeros(N_samples)
omegas = np.ones(N_samples)