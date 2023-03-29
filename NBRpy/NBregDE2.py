import pandas as pd
from patsy import dmatrices, dmatrix
import numpy as np
from scipy.special import digamma
from NBreg_VB import NBreg_VB
from CRP import CRT
from pypolyagamma import PyPolyaGamma
from utils import multi_pgdraw, CRT, logOnePlusExp
from numpy import linalg as LA

dfPexorig_wide = pd.read_csv("dfPexorig_wide.csv")
counts = np.array(dfPexorig_wide.iloc[:, 3:])

X = np.ones((6, 2))
X[:3, 1] = 0
X = np.transpose(X)
counts = np.array(dfPexorig_wide.iloc[:, 3:])

Collections = 1000
Burnin = 1000

y = counts
dd = np.shape(y)[0]  # K wise
non_zero_rows = np.where(np.sum(y, axis=1) != 0)
y = y[non_zero_rows]

ngenes = np.shape(y)[0]
nsamples = np.shape(y)[1]

a0 = 0.01
b0 = 0.01
c0 = 0.01
d0 = 0.01
e0 = 0.01
f0 = 0.01
g0 = 0.01

P = np.shape(X)[0]  # rows samples for the design matrix
beta = np.zeros((P, ngenes))  # 4
r = np.repeat(100, nsamples)
h = 1
alpha = np.repeat(100, P)
Beta, sigmak, r, h, alpha = NBreg_VB(counts, X, 100)

Psi = np.zeros((ngenes, nsamples))
XYT = np.dot(X, np.transpose(y))
beta_mean = np.zeros((P, ngenes))
beta2_samples = np.zeros((ngenes, Collections))
beta3_samples = np.zeros((ngenes, Collections))
tehta_mean = np.zeros((ngenes, 2 * Collections))
iterMax = Burnin + Collections
yy = np.where(y > 10000, 10000, y)
for iter in range(iterMax):
    Psi = np.dot(np.transpose(Beta), X)
    # Sample r_j
    ell = np.repeat(0, nsamples)
    for z in range(nsamples):
        yr = y[y[:, z] > 10000, z]
        if yr.size == 0:
            yr = 0
        part_one = np.random.poisson(1, r[z] * np.sum(digamma(yy + r[z])) - digamma(10000 + r[z]))
        ell[z] = CRT(yy[:, z], r[z])
    r = np.random.gamma(scale=h + np.sum(logOnePlusExp(Psi), axis=0), shape=a0 + ell, size=nsamples)
    # Sample alpha
    alpha = np.random.gamma(size=P, shape=c0 + ngenes / 2, scale=d0 + 0.5 * np.sum(Beta ** 2, 1))
    # Smple h
    h = np.random.gamma(size=1, scale=b0 + nsamples * a0, shape=g0 + sum(r))
    # np.random.poisson(1,r[j]*np.sum(digamma(yr+r[j]) - digamma(10000+r[j])))

    # Sample omega
    temp = y + np.repeat(r, ngenes).reshape((ngenes, nsamples))
    pg = PyPolyaGamma()
    omega = multi_pgdraw(pg, np.ravel(temp), np.ravel(Psi)).reshape((ngenes, nsamples))
    if np.any(np.isnan(omega)):
        omega[np.isnan(omega)] = np.mean(np.nanmean(omega))

    # omega = multi_pgdraw(pg, np.ravel(Psi), np.ravel(temp)).reshape((ngenes,nsamples))

    # Sample Beta, phi and Psi
    A = np.diag(alpha)
    for k in range(ngenes - 1):
        _, v = LA.eig(A + X @ np.diag(omega[k,]) @ np.transpose(X))
        if np.any(v < 0):
            continue
        else:
            temp = np.linalg.cholesky(A + X @ np.diag(omega[k,]) @ np.transpose(X))
            Beta[:, k] = temp @ (np.random.normal(0, 1, P) + np.transpose(temp)) @ (0.5 * (XYT[:, k] - X @ r))
    if iter > Burnin:
        beta_mean = beta_mean + Beta
        tehta_mean[:, iter - Burnin] = np.exp(Beta[1, :])
    # two conditions #later






