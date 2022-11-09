from copy import deepcopy
import sys
import numpy as np

from utils import train_test_split


class LinearLayer:
    def __init__(self, in_dim, out_dim, rng) -> None:

        self.rng = rng

        self.params = dict()
        self.params["W"] = self.rng.normal(loc=0.0, scale=0.1, size=(in_dim, out_dim))
        self.params["b"] = self.rng.normal(loc=0.0, scale=0.1, size=(1, out_dim))

        self.gradient = dict()
        self.gradient["W"] = np.zeros(shape=(in_dim, out_dim))
        self.gradient["b"] = np.zeros(shape=(1, out_dim))

    def forward(self, X):
        return np.dot(X, self.params["W"]) + self.params["b"]

    def backward(self, X, grad):
        self.gradient["W"] = np.matmul(X.T, grad)
        self.gradient["b"] = np.array([np.sum(grad, axis=0)])
        return np.matmul(grad, self.params["W"].T)


class ReLU:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, X):
        return np.maximum(0, X)

    def backward(self, X, grad):
        return np.multiply(grad, np.where(X <= 0, 0, 1))


class Dropout:
    def __init__(self, r, rng) -> None:
        self.r = r
        self.rng = rng
        self.mask = None

    def forward(self, X, is_train):
        if is_train:
            self.mask = (self.rng.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0 - self.r))
        else:
            self.mask = np.ones(X.shape)
        return np.multiply(X, self.mask)

    def backward(self, X, grad):
        return np.multiply(self.mask, grad)


class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.zeros(X.shape).reshape(-1)
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0
        self.expand_Y = self.expand_Y.reshape(X.shape)

        self.calib_logit = X - np.amax(X, axis=1, keepdims=True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis=1, keepdims=True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit

        forward_out = (
            -np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        )
        return forward_out

    def backward(self, X, Y):
        backward_output = -(self.expand_Y - self.prob) / X.shape[0]
        return backward_output


def add_momentum(model):
    momentum = dict()
    for module_name, module in model.items():
        if hasattr(module, "params"):
            for key, _ in module.params.items():
                momentum[module_name + "_" + key] = np.zeros(shape=module.gradient[key].shape)
    return momentum


def predict_label(f):
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))


class DataBatchMaker:
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        self.N, self.d = self.X.shape

    def get(self, idxs):
        X_batch = np.zeros((len(idxs), self.d))
        y_batch = np.zeros((len(idxs), 1))
        for i in range(len(idxs)):
            X_batch[i] = self.X[idxs[i]]
            y_batch[i, :] = self.y[idxs[i]]
        return X_batch, y_batch


def gradient_descent(model, momentum, alpha, learning_rate):
    for module_name, module in model.items():
        if hasattr(module, "params"):
            for key, _ in module.params.items():
                g = module.gradient[key]
                if alpha <= 0.0:
                    module.params[key] = module.params[key] - np.multiply(learning_rate, g)
                else:
                    momentum[f"{module_name}_{key}"] = np.multiply(
                        alpha, momentum[f"{module_name}_{key}"]
                    ) - np.multiply(learning_rate, g)
                    module.params[key] = module.params[key] + momentum[f"{module_name}_{key}"]
    return model


def train(X, y, val_ratio, model, n_epoch, mini_batch_size, alpha, learning_rate, step, rng):
    X_train, y_train, X_val, y_val = train_test_split(X, y, test_ratio=val_ratio, rng=rng)

    n_train, _ = X_train.shape
    n_val, _ = X_val.shape

    train_set = DataBatchMaker(X_train, y_train)
    val_set = DataBatchMaker(X_val, y_val)

    # Momentum
    if alpha > 0.0:
        momentum = add_momentum(model)
    else:
        momentum = None

    best_val_loss = sys.maxsize
    best_model = None
    best_epoch = 0

    for n in range(n_epoch):

        if (n % step == 0) and (n != 0):
            learning_rate = learning_rate * 0.1

        idx_order = rng.permutation(n_train)

        train_acc = 0.0
        train_loss = 0.0
        train_count = 0

        val_acc = 0.0
        val_count = 0
        val_loss = 0.0

        for i in range(int(np.floor(n_train / mini_batch_size))):

            # get a mini-batch of data
            x_train, y_train = train_set.get(idx_order[i * mini_batch_size : (i + 1) * mini_batch_size])

            # Forward Pass
            a1 = model["FC_1"].forward(x_train)
            h1 = model["RELU_1"].forward(a1)
            a2 = model["FC_2"].forward(h1)
            h2 = model["RELU_2"].forward(a2)
            a3 = model["FC_3"].forward(h2)
            h3 = model["RELU_3"].forward(a3)
            d1 = model["DO_1"].forward(h3, is_train=True)
            a4 = model["FC_4"].forward(d1)
            loss = model["LOSS"].forward(a4, y_train)

            # Backward Pass
            grad_a4 = model["LOSS"].backward(a4, y_train)
            grad_d1 = model["FC_4"].backward(d1, grad_a4)
            grad_h3 = model["DO_1"].backward(h3, grad_d1)
            grad_a3 = model["RELU_3"].backward(a3, grad_h3)
            grad_h2 = model["FC_3"].backward(h2, grad_a3)
            grad_a2 = model["RELU_2"].backward(a2, grad_h2)
            grad_h1 = model["FC_2"].backward(h1, grad_a2)
            grad_a1 = model["RELU_1"].backward(a1, grad_h1)
            _ = model["FC_1"].backward(x_train, grad_a1)

            # Update gradient
            model = gradient_descent(model, momentum, alpha, learning_rate)

        # Train Accuracy
        for i in range(int(np.floor(n_train / mini_batch_size))):

            x_train, y_train = train_set.get(np.arange(i * mini_batch_size, (i + 1) * mini_batch_size))

            loss, preds = calculate_loss(model, x_train, y_train)
            train_loss += loss
            train_acc += np.sum(preds == y_train)
            train_count += len(y_train)

        train_acc = train_acc / train_count

        # Validation Accuracy
        for i in range(int(np.floor(n_val / mini_batch_size))):
            x_val, y_val = val_set.get(np.arange(i * mini_batch_size, (i + 1) * mini_batch_size))
            loss, preds = calculate_loss(model, x_val, y_val)
            val_loss += loss
            val_acc += np.sum(preds == y_val)
            val_count += len(y_val)

        val_acc = val_acc / val_count

        print(
            f"Epoch: {n + 1}, Training Loss: {train_loss}, Training Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}"
        )

        # Store best model
        if val_loss < best_val_loss:
            best_epoch = n
            best_model = deepcopy(model)
            best_val_loss = val_loss

    return best_epoch, best_model


def predict(model, X):
    a1 = model["FC_1"].forward(X)
    h1 = model["RELU_1"].forward(a1)
    a2 = model["FC_2"].forward(h1)
    h2 = model["RELU_2"].forward(a2)
    a3 = model["FC_3"].forward(h2)
    h3 = model["RELU_3"].forward(a3)
    d1 = model["DO_1"].forward(h3, is_train=False)
    a4 = model["FC_4"].forward(d1)
    labels = np.squeeze(predict_label(a4))
    return labels


def calculate_loss(model, x, y):
    a1 = model["FC_1"].forward(x)
    h1 = model["RELU_1"].forward(a1)
    a2 = model["FC_2"].forward(h1)
    h2 = model["RELU_2"].forward(a2)
    a3 = model["FC_3"].forward(h2)
    h3 = model["RELU_3"].forward(a3)
    d1 = model["DO_1"].forward(h3, is_train=False)
    a4 = model["FC_4"].forward(d1)
    loss = model["LOSS"].forward(a4, y)

    return loss, predict_label(a4)
