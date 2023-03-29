import random
import sys
import numpy as np
from numpy.linalg import inv
import numpy.random as npr
from pypolyagamma import PyPolyaGamma


def logOnePlusExp(x):
    y = np.zeros_like(x)
    y[x < 0] = np.log1p(np.exp(x[x < 0]))
    y[x >= 0] = x[x >= 0] + np.log1p(np.exp(-x[x >= 0]))
    return y


def CRT(x,r):
    large_number = sys.maxsize # 232 − 1
    Lsum = 0
    maxx = max(x)
    prob = [None]*(max(x)-1)
    for i in range(max(x)-1):
        prob[i] = r/(r+i)
    for z in range(len(x)-1):
        for j in range(x[z]-1):
            if random.randrange(0, sys.maxsize, 2) <= (prob[j]*large_number):
                Lsum += 1
    return Lsum


def sigmoid(x):
    """Numerically stable sigmoid function.
    """
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def multi_pgdraw(pg, B, C):
    """Utility function for calling `pgdraw` on every pair in vectors B, C.
    """
    arr = np.array([pg.pgdraw(b, c) for b, c in zip(B, C)])
    return arr

def gen_bimodal_data(N, p):
    """Generate bimodal data for easy sanity checking.
    """
    y     = npr.random(N) < p
    X     = np.empty(N)
    X[y]  = npr.normal(0, 1, size=y.sum())
    X[~y] = npr.normal(4, 1.4, size=(~y).sum())
    return X, y.astype(int)