MLP_HYPERPARAMETERS = {
    'module__n_layers': Integer(2, 5, default=3),
    'module__d_layers': Integer(32, 256, default=64),
    'module__d_embedding': Integer(4, 16, default=8),
    'module__dropout': Real(0.1, 0.5, default=0.3),
    'optimizer__weight_decay': Real(1e-5, 1e-3, 'log-uniform', default=1e-4),
    'max_epochs': Integer(50, 200, default=100),
    'batch_size': Integer(32, 256, default=128),
    'callbacks__early_stopping__patience': Integer(5, 20, default=10),
    'callbacks__scheduler__T_max': Integer(100, 500, default=200),

    'optimizer__param_groups': Categorical([
        [
            {'params': ['category_embeddings.*'], 'lr': Real(0.005, 0.05, default=0.01)},
            {'params': ['net.*'], 'lr': Real(0.0005, 0.005, default=0.001)},
            {'params': ['head.*'], 'lr': Real(0.00005, 0.0005, default=0.0001)}
        ]
    ])
}

SVM_HYPERPARAMETERS = {
    'C': Real(0.1, 1000, prior='log-uniform'),
    'kernel': Categorical([
        {'name': 'rbf', 'sigma': Real(0.01, 10, prior='log-uniform')},
        {'name': 'poly', 'degree': 3, 'coef0': Real(0, 5)}
    ]),
    'toler': 0.001,
    'maxIter': 1000
}

XGBOOST_HYPERPARAMETERS = {
    'max_depth': Integer(3, 10, default=6),
    'gamma': Real(0, 5, default=0),
    'eta': Real(0.01, 0.3, 'log-uniform', default=0.3),
    'subsample': Real(0.5, 1.0, default=0.8),
    'colsample_bytree': Real(0.5, 1.0, default=0.8),
    'num_round': Integer(100, 500, default=300),
    'focal_gamma': Real(1.0, 5.0, default=2.0),
    'lambda': Real(1e-3, 10, 'log-uniform', default=1),
    'alpha': Real(1e-3, 10, 'log-uniform', default=0)
}


TABTRANSFORMER_HYPERPARAMETERS = {
    'n_layers': Integer(3, 8, default=6),
    'd_token': Integer(32, 512, default=128),
    'n_heads': Integer(4, 12, default=8),
    'd_ffn_factor': Real(2.0, 6.0, default=4.0),

    'kv_compression': Real(0.1, 0.5, default=None),
    'attention_dropout': Real(0.0, 0.5, default=0.2),

    'residual_dropout': Real(0.0, 0.3, default=0.1),
    'ffn_dropout': Real(0.0, 0.5, default=0.3),
    'weight_decay': Real(1e-6, 1e-2, 'log-uniform', default=1e-4),


    'batch_size': Integer(128, 2048, default=1024),
    'lr': Real(1e-5, 1e-2, 'log-uniform', default=1e-3),
    'patience': Integer(5, 20, default=10),

    'cat_min_frequency': Real(0.0, 0.1, default=0.01),
    'token_bias': Categorical([True, False], default=True)
}