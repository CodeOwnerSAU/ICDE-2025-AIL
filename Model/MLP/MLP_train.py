import math
import typing as ty
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from Model.MLP.EnhanceMLP import EnhancedMLP
from Model.MLP.LoadData import load_training_data


class InputShapeSetter(Callback):
    def on_train_begin(self, net, X, y):
        n_features = X.shape[1]
        net.set_params(
            module__d_in=n_features,
            module__n_classes=len(np.unique(y))
        )

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_training_data()


    categorical_indicator = np.array([False] * 8 + [True])  # 8个数值特征+1个分类特征


    net = NeuralNetClassifier(
        module=EnhancedMLP,
        module__n_layers=3,
        module__d_layers=64,
        module__dropout=0.3,
        module__d_embedding=8,
        module__categories=[3],
        module__categorical_indicator=categorical_indicator,
        criterion=nn.CrossEntropyLoss,


        optimizer=torch.optim.Adam([
            {'params': ['category_embeddings.*'], 'lr': 0.01},
            {'params': ['net.*'], 'lr': 0.001},
            {'params': ['head.*'], 'lr': 0.0001}
        ], weight_decay=1e-4),

        callbacks=[
            InputShapeSetter(),
            skorch.callbacks.EarlyStopping(patience=10),
            ('scheduler', CosineAnnealingLR(T_max=200))
        ],


        device='cuda' if torch.cuda.is_available() else 'cpu',
        train_split=None,
        iterator_train__shuffle=True
    )


    scaler = GradScaler()

    print("Starting training...")
    for epoch in range(100):
        net.partial_fit(X_train, y_train, epochs=1)

        net.callbacks_[2][1].step()

    with torch.no_grad():
        y_pred = net.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"\nTest Accuracy: {accuracy:.4f}")


    # print("\nSample predictions:")
    # print("True labels:", y_test[:5])
    # print("Pred labels:", y_pred[:5])
