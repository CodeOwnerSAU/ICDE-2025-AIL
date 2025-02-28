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