
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
from Model.XGBoost.Bayesian import xgb_bayesian_optimization
from Model.XGBoost.LoadData import load_training_data
from Model.XGBoost.MutiClassXGBoost import MultiClassXGBoost


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
