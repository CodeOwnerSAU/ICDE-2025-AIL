import math
import typing as ty
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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