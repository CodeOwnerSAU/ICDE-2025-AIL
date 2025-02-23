import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

param_space = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 5),
    'reg_alpha': Real(0, 1),
    'reg_lambda': Real(0, 1),
    'scale_pos_weight': Real(0.5, 2)
}


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


class EnhancedXGBClassifier(XGBClassifier):
    def fit(self, X, y, **kwargs):
        return super().fit(
            X, y,
            early_stopping_rounds=20,
            eval_metric='mlogloss',
            eval_set=[(X_val, y_val)],
            verbose=False,
            **kwargs
        )


def xgb_multiclass_optimized():
    X_train, y_train, X_test, y_test = load_training_data()

    opt = BayesSearchCV(
        estimator=EnhancedXGBClassifier(
            objective='multi:softmax',
            num_class=len(np.unique(y_train)),
            tree_method='hist',
            enable_categorical=True,
            use_label_encoder=False
        ),
        search_spaces=param_space,
        scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=5, shuffle=True),
        n_iter=500,
        n_jobs=-1,
        verbose=1
    )

    opt.fit(X_train, y_train)

    best_model = opt.best_estimator_

    y_pred = best_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    feature_importances = pd.Series(
        best_model.feature_importances_,
        index=num_features + list(poi_train.columns)
    )

    print(feature_importances.sort_values(ascending=False)[:10])

    return best_model


model = xgb_multiclass_optimized()