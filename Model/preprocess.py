
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from skopt.space import Real, Categorical
import sys

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