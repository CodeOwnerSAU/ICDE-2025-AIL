from Model.XGBoost.FocalMultiloss import FocalMultiLoss
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

class MultiClassXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, num_class=3, num_round=100, max_depth=6, eta=0.3,
                 gamma=0, subsample=0.8, colsample_bytree=0.8,
                 objective='multi:softmax', focal_gamma=None):
        self.num_class = num_class
        self.num_round = num_round
        self.max_depth = max_depth
        self.eta = eta
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.focal_gamma = focal_gamma
        self.model = None

    def fit(self, X, y, eval_set=None):
        params = {
            'max_depth': self.max_depth,
            'eta': self.eta,
            'gamma': self.gamma,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'objective': self.objective,
            'num_class': self.num_class,
            'eval_metric': 'mlogloss'
        }

        dtrain = xgb.DMatrix(X, label=y)

        if self.focal_gamma is not None:
            focal_loss = FocalMultiLoss(gamma=self.focal_gamma, num_class=self.num_class)
            self.model = xgb.train(params, dtrain, self.num_round,
                                   obj=focal_loss.focal_multi_object)
        else:
            self.model = xgb.train(params, dtrain, self.num_round)

        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest).astype(int)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        raw_preds = self.model.predict(dtest, output_margin=True)
        return softmax(raw_preds.reshape(-1, self.num_class))

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)