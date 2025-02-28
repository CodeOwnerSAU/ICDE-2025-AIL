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


class FocalMultiLoss:
    def __init__(self, gamma=2.0, num_class=10):
        self.gamma = gamma
        self.num_class = num_class

    def focal_multi_object(self, preds, dtrain):
        labels = dtrain.get_label().astype(int)
        probs = softmax(preds.reshape(-1, self.num_class), axis=1)


        rows = np.arange(len(labels))
        p_true = probs[rows, labels]
        focal_loss = - (1 - p_true) ** self.gamma * np.log(p_true + 1e-9)


        grad = probs.copy()
        grad[rows, labels] -= 1
        grad *= (1 - p_true) ** self.gamma
        grad = grad.ravel()


        hess = 2 * probs * (1 - probs) * (1 - p_true) ** self.gamma
        hess = hess.ravel()

        return grad, hess


def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


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


def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    # plt.figure(figsize=(10, 8))
    # for i in range(model.num_class):
    #     fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
    #     roc_auc = auc(fpr, tpr)
    #     plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":

    X_train, y_train, X_test, y_test, class_names = load_training_data()
    num_class = len(class_names)

    print("Starting Bayesian Optimization...")
    best_params = xgb_bayesian_optimization(X_train, y_train, num_class)
    print("\nBest parameters found:")
    print(best_params)

    final_model = MultiClassXGBoost(
        num_class=num_class,
        max_depth=int(best_params['max_depth']),
        eta=best_params['eta'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        focal_gamma=2.0
    )

    final_model.fit(X_train, y_train)

    evaluate_model(final_model, X_test, y_test, class_names)

    xgb.plot_importance(final_model.model)
    plt.title('Feature Importance')
    plt.show()
