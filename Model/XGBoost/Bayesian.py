import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def xgb_bayesian_optimization(X, y, num_class, init_points=5, n_iter=20):
    dtrain = xgb.DMatrix(X, label=y)

    def xgb_cv(max_depth, eta, subsample, colsample_bytree, gamma):
        params = {
            'max_depth': int(max_depth),
            'eta': max(eta, 0.01),
            'subsample': max(min(subsample, 1), 0.1),
            'colsample_bytree': max(min(colsample_bytree, 1), 0.1),
            'gamma': max(gamma, 0),
            'objective': 'multi:softmax',
            'num_class': num_class,
            'eval_metric': 'mlogloss'
        }

        cv_results = xgb.cv(
            params, dtrain,
            num_boost_round=300,
            nfold=5,
            stratified=True,
            early_stopping_rounds=20,
            verbose_eval=False
        )
        return -cv_results['test-mlogloss-mean'].iloc[-1]

    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds={
            'max_depth': (3, 10),
            'eta': (0.01, 0.3),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.5, 1),
            'gamma': (0, 5)
        },
        random_state=42
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer.max['params']