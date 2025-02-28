import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler


def load_training_data():

    train_data = pd.read_csv("AIL_peak_train.csv")
    test_data = pd.read_csv("AIL_peak_test.csv")

    poi_train = csr_matrix(pd.get_dummies(train_data['POI_distrubtion'], prefix='POI'))
    poi_test = csr_matrix(pd.get_dummies(test_data['POI_distrubtion'], prefix='POI')
                          .reindex(columns=poi_train.columns, fill_value=0))


    num_features = ['Road_Distance', 'Query_Density', 'Execution_Time',
                    'Keyword_Count', 'POI_Density', 'POI_Type', 'POI_Contain']
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_data[num_features])
    X_test_num = scaler.transform(test_data[num_features])


    X_train = csr_matrix(np.hstack([X_train_num, poi_train.toarray()]))
    X_test = csr_matrix(np.hstack([X_test_num, poi_test.toarray()]))
    y_train = train_data['Label'].values
    y_test = test_data['Label'].values

    return X_train, y_train, X_test, y_test
