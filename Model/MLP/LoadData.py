import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


def load_training_data():

    train_data = pd.read_csv("../../AIL_peak_train.csv")
    test_data = pd.read_csv("../../AIL_peak_test.csv")
    le = LabelEncoder()
    train_data['POI_distribution'] = le.fit_transform(train_data['POI_distrubtion'])
    test_data['POI_distribution'] = le.transform(test_data['POI_distrubtion'])

    numerical_features = [
        'Road_Distance', 'Keyword_Count', 'POI_Density',
        'Query_Density', 'POI_Type', 'POI_Contain', 'Execution_Time',
    ]
    categorical_features = ['POI_distribution']


    scaler = RobustScaler(quantile_range=(5, 95))
    X_train_num = scaler.fit_transform(train_data[numerical_features])
    X_test_num = scaler.transform(test_data[numerical_features])


    X_train = np.hstack([
        X_train_num,
        train_data[categorical_features].values.astype(np.float32)
    ])
    X_test = np.hstack([
        X_test_num,
        test_data[categorical_features].values.astype(np.float32)
    ])


    y_train = train_data['Label'].values.astype(np.int64)
    y_test = test_data['Label'].values.astype(np.int64)

    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test

