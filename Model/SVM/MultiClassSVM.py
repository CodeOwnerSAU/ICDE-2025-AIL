from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.utils.extmath import softmax
from sklearn.utils.parallel import Parallel, delayed

from Model.SVM.PlattSMO import PlattSMO



class MultiClassSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, toler=0.001, maxIter=1000,
                 kernel={'name': 'rbf', 'sigma': 1.0}, n_jobs=-1):
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.kernel = kernel
        self.n_jobs = n_jobs
        self.classifiers_ = {}
        self.classes_ = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.classes_ = np.unique(y)
        self.pairs_ = list(combinations(self.classes_, 2))


        self.classifiers_ = dict(zip(
            self.pairs_,
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._train_binary)(X, y, cls1, cls2)
                for cls1, cls2 in self.pairs_
            )
        ))
        return self

    def _train_binary(self, X, y, cls1, cls2):
        mask = np.isin(y, [cls1, cls2])
        X_sub = X[mask]
        y_sub = np.where(y[mask] == cls1, 1, -1)


        weights = compute_class_weight('balanced', classes=[-1, 1], y=y_sub)
        svm = PlattSMO(X_sub, y_sub, self.C * weights[1], self.toler,
                       self.maxIter, class_weights=weights,
                       **self.kernel)
        svm.smoP()
        return svm

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        decision = np.zeros((X.shape[0], len(self.classes_)))

        for (cls1, cls2), clf in self.classifiers_.items():
            pred = clf.predict(X, raw_output=True)
            cls1_idx = np.where(self.classes_ == cls1)[0][0]
            cls2_idx = np.where(self.classes_ == cls2)[0][0]
            decision[:, cls1_idx] += pred
            decision[:, cls2_idx] -= pred

        return softmax(decision, axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]