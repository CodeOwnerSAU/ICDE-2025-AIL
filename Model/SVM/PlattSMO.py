import numpy as np


class PlattSMO:
    def __init__(self, dataMat, classlabels, C, toler, maxIter, class_weights=None, **kernelargs):
        self.X = np.asarray(dataMat, dtype=np.float64)
        self.y = np.where(np.asarray(classlabels) == 1, 1, -1).flatten()
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.m, self.n = self.X.shape
        self.alpha = np.zeros(self.m, dtype=np.float64)
        self.b = 0.0
        self.eCache = np.zeros((self.m, 2))
        self.kernelargs = kernelargs
        self.class_weights = class_weights if class_weights is not None else np.ones(2)
        self._init_kernel_matrix()

    def _init_kernel_matrix(self):

        X = self.X
        kernel_type = self.kernelargs['name']

        if kernel_type == 'rbf':
            gamma = 1 / (2 * self.kernelargs['sigma'] ** 2)
            pairwise_dists = np.sum(X ** 2, axis=1)[:, None] - 2 * np.dot(X, X.T) + np.sum(X ** 2, axis=1)
            self.K = np.exp(-gamma * pairwise_dists)
        elif kernel_type == 'linear':
            self.K = np.dot(X, X.T)
        elif kernel_type == 'poly':
            degree = self.kernelargs.get('degree', 3)
            coef0 = self.kernelargs.get('coef0', 1.0)
            self.K = (coef0 + np.dot(X, X.T)) ** degree

    def kernelTrans(self, x, z):

        kernel_type = self.kernelargs['name']
        if kernel_type == 'linear':
            return np.dot(x, z)
        elif kernel_type == 'rbf':
            gamma = 1.0 / (2 * self.kernelargs['sigma'] ** 2)
            return np.exp(-gamma * np.linalg.norm(x - z) ** 2)
        elif kernel_type == 'poly':
            degree = self.kernelargs.get('degree', 3)
            coef0 = self.kernelargs.get('coef0', 1.0)
            return (coef0 + np.dot(x, z)) ** degree
        else:
            raise ValueError("Unsupported kernel type")

    def calcEK(self, k):
        return np.dot(self.alpha * self.y, self.K[:, k]) + self.b - self.y[k]

    def updateEK(self, k):
        self.eCache[k] = [1, self.calcEK(k)]