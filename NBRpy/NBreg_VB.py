import pandas as pd
from patsy import dmatrices, dmatrix
import numpy as np
from scipy.special import digamma

'''
example for running the code:

#Tables
dfPexorig_wide = pd.read_csv("dfPexorig_wide.csv")
counts = np.array(dfPexorig_wide.iloc[:,3:])
#design matrix
X = np.ones((6,2))
X[:3,1] = 0
X = np.transpose(X)

'''

def NBreg_VB(counts, X, maxIter = 1000):
    y = counts
    dd = np.shape(y)[0]  # K wise
    non_zero_rows = np.where(np.sum(y, axis=1) != 0)
    y = y[non_zero_rows]
    K = np.shape(y)[0]  # rows Genes
    J = np.shape(y)[1]  # colums Samples
    P = np.shape(X)[0]  # rows samples for the design matrix
    a0 = 0.01
    b0 = 0.01
    c0 = 0.01
    d0 = 0.01
    e0 = 0.01
    f0 = 0.01
    g0 = 0.01

    rj = np.ones((J))
    aj = np.ones((J))
    hj = np.ones((J))
    # 6 samples
    cp = np.ones((P))
    dp = np.ones((P))  # 6 samples acording to the design matrix
    cp = (c0 + K / 2) * cp
    muk = np.zeros((P, K))  # matrix of 6 rows 5 columns
    sigmak = np.zeros((4, 2, 2))
    b = a0 + b0
    g = 1
    # mean quantities
    Elogr = digamma(aj) - np.log(hj)
    part_one = np.ones((K, 1)) * rj
    part_two = digamma(y + np.ones((K, 1)) * rj)
    part_three = np.ones((K, 1)) * digamma(np.transpose(digamma(rj)))
    El = (part_one * part_two) - part_three
    Er = (aj) / (hj)
    Ealpha = (cp) / (dp)
    Eh = (b) / (g)
    Elog1pexp = np.zeros((K, J))  # 4
    Ew = np.zeros((K, J))

    def logOnePlusExp(x):
        y = np.zeros_like(x)
        y[x < 0] = np.log1p(np.exp(x[x < 0]))
        y[x >= 0] = x[x >= 0] + np.log1p(np.exp(-x[x >= 0]))
        return y

    for k in range(K):
        betak = np.random.normal(0, 1, P)
        Elog1pexp[k,] = logOnePlusExp(np.dot(betak, X))
        Ew[k,] = (y[k,] + Er) * (np.tanh(np.dot(betak, X / 2)) / (2 * np.dot(betak, X)))
    for iter in range(maxIter):
        # update L_{jk}
        rj = np.exp(Elogr)
        aj = np.sum(El, axis=0)
        hj = Eh + np.sum(Elog1pexp, axis=0)
        Er = (aj) / (hj)
        Elogr = digamma(aj) - np.log(hj)
        # beta K
        temp = 0

        # start loop
        for k in range(K):
            sigmak[k, :, :] = np.linalg.solve(np.dot(np.dot(X, np.diag(Ew[k,])), np.transpose(X)) + np.diag(Ealpha),
                                              np.diag(np.ones(P)))
            muk[:, k] = np.transpose(np.dot(sigmak[k], np.dot(0.5 * X, (y[k,] - Er[k]))))
            # betak <- mvrnorm(n = 500, mu = muk[,k], Sigma = sigmak[[k]], tol = 1e-30)

            betak = np.random.multivariate_normal(mean=muk[:, k], cov=sigmak[k, :, :], size=500)  # 500 by P
            Elog1pexp[k,] = np.mean(logOnePlusExp(np.dot(betak, X)), axis=0)
            Ew[k,] = (y[k,] + Er) * np.mean((np.tanh(np.dot(betak, X / 2)) / (logOnePlusExp(2 * np.dot(betak, X)))), axis=0)
            temp = temp + np.diag(sigmak[k, :, :]) + muk[:, k] ** 2
        dp = d0 + 0.5 * temp
        Ealpha = cp / dp
        # h
        g = g0 + sum(Er)
        Eh = b / g
    return muk, sigmak, Er, Eh, Ealpha
