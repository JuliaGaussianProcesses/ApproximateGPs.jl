import numpy as np
import gpflow

Ntrain = 5000
M = 300
maxiter = 100

fn = "CASP.csv"  # protein dataset via bayesian_benchmarks
dat = np.loadtxt(fn, skiprows=1, delimiter=",")
Y = dat[:, :1]
X = dat[:, 1:]
Xs = (X - X.mean(0)) / X.std(0)
Ys = (Y - Y.mean(0)) / Y.std(0)
Xtrain = Xs[:Ntrain]
Ytrain = Ys[:Ntrain]
np.random.seed(12345)
idxZ = np.random.permutation(len(Xtrain))[:M]
Z = Xtrain[idxZ]

k = gpflow.kernels.SquaredExponential()
m = gpflow.models.SVGP(k, gpflow.likelihoods.Gaussian(0.1), Z)
gpflow.set_trainable(m.inducing_variable, False)

loss = m.training_loss_closure((Xtrain, Ytrain))

print("LOSS:", loss())

import time
t0 = time.time()
gpflow.optimizers.Scipy().minimize(loss, m.trainable_variables, options=dict(maxiter=maxiter))
t1 = time.time()

