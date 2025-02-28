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
def load_training_data():
    train_data = pd.read_csv("AIL_peak_train.csv")
    test_data = pd.read_csv("AIL_peak_test.csv")


    poi_train = pd.get_dummies(train_data['POI_distrubtion'], prefix='POI')
    poi_test = pd.get_dummies(test_data['POI_distrubtion'], prefix='POI')
    poi_test = poi_test.reindex(columns=poi_train.columns, fill_value=0)


    num_features = ['Road_Distance', 'Query_Density', 'Execution_Time',
                    'Keyword_Count', 'POI_Density', 'POI_Type', 'POI_Contain']
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_data[num_features])
    X_test_num = scaler.transform(test_data[num_features])


    X_train = csr_matrix(hstack([X_train_num, poi_train]))
    X_test = csr_matrix(hstack([X_test_num, poi_test]))


    le = LabelEncoder()
    y_train = le.fit_transform(train_data['Label'])
    y_test = le.transform(test_data['Label'])

    return X_train, y_train, X_test, y_test, le.classes_