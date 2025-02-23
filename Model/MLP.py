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


def load_training_data():

    train_data = pd.read_csv("AIL_peak_train.csv")
    test_data = pd.read_csv("AIL_peak_test.csv")


    train_data['POI_Interaction'] = train_data['POI_Density'] * train_data['POI_Type']
    test_data['POI_Interaction'] = test_data['POI_Density'] * test_data['POI_Type']


    le = LabelEncoder()
    train_data['POI_distribution'] = le.fit_transform(train_data['POI_distrubtion'])
    test_data['POI_distribution'] = le.transform(test_data['POI_distrubtion'])


    numerical_features = [
        'Road_Distance', 'Keyword_Count', 'POI_Density',
        'Query_Density', 'POI_Type', 'POI_Contain', 'Execution_Time',
        'POI_Interaction'
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



class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class EnhancedMLP(nn.Module):
    def __init__(
            self,
            d_in: int,
            n_layers: int,
            d_layers: int,
            dropout: float,
            n_classes: int,
            categories: ty.List[int],
            d_embedding: int,
            categorical_indicator: np.ndarray
    ):
        super().__init__()
        self.categorical_indicator = categorical_indicator

        self.scale_layer = nn.Parameter(torch.ones(d_in))

        if categories:
            self.category_embeddings = nn.Embedding(
                num_embeddings=categories[0],
                embedding_dim=d_embedding
            )
            nn.init.kaiming_normal_(self.category_embeddings.weight)
            d_in = d_in - 1 + d_embedding
        else:
            self.category_embeddings = None


        self.gate_layers = nn.ModuleList([
            nn.Linear(d_layers, d_layers) for _ in range(n_layers)
        ])


        layers = []
        for i in range(n_layers):
            in_dim = d_layers if i > 0 else d_in
            layers.extend([
                nn.Linear(in_dim, d_layers),
                Mish(),
                nn.Dropout(dropout)
            ])
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(d_layers, n_classes)

    def forward(self, x):
        x = x * self.scale_layer
        x_num = x[:, ~self.categorical_indicator].float()
        x_cat = x[:, self.categorical_indicator].long()
        if self.category_embeddings is not None:
            embeddings = self.category_embeddings(x_cat[:, 0])
            x = torch.cat([x_num, embeddings], dim=1)
        else:
            x = x_num

        for i, layer in enumerate(self.net):
            x = layer(x)
            if isinstance(layer, Mish):

                gate = torch.sigmoid(self.gate_layers[i // 3](x))
                x = x * gate
        return self.head(x)


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
