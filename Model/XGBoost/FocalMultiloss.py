import numpy as np


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