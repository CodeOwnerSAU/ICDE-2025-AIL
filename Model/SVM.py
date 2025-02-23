import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import sys


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


def load_training_data():

    train_data = pd.read_csv("AIL_peak_train.csv")
    test_data = pd.read_csv("AIL_peak_test.csv")


    for df in [train_data, test_data]:
        df['POI_Interaction'] = df['POI_Density'] * df['POI_Type']


    poi_train = csr_matrix(pd.get_dummies(train_data['POI_distrubtion'], prefix='POI'))
    poi_test = csr_matrix(pd.get_dummies(test_data['POI_distrubtion'], prefix='POI')
                          .reindex(columns=poi_train.columns, fill_value=0))


    num_features = ['Road_Distance', 'Query_Density', 'Execution_Time',
                    'Keyword_Count', 'POI_Density', 'POI_Type', 'POI_Contain',
                    'POI_Interaction']
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_data[num_features])
    X_test_num = scaler.transform(test_data[num_features])


    X_train = csr_matrix(np.hstack([X_train_num, poi_train.toarray()]))
    X_test = csr_matrix(np.hstack([X_test_num, poi_test.toarray()]))
    y_train = train_data['Label'].values
    y_test = test_data['Label'].values

    return X_train, y_train, X_test, y_test


def main():

    X_train, y_train, X_test, y_test = load_training_data()

    param_space = {
        'C': Real(0.1, 1000, prior='log-uniform'),
        'kernel': Categorical([
            {'name': 'rbf', 'sigma': Real(0.01, 10, prior='log-uniform')},
            {'name': 'poly', 'degree': 3, 'coef0': Real(0, 5)}
        ])
    }

    bayes_search = BayesSearchCV(
        estimator=MultiClassSVM(n_jobs=4),
        search_spaces=param_space,
        n_iter=50,
        cv=3,
        n_jobs=1,
        random_state=42,
        scoring='accuracy'
    )
    bayes_search.fit(X_train, y_train)

    best_svm = bayes_search.best_estimator_
    y_pred = best_svm.predict(X_test)



if __name__ == "__main__":
    main()