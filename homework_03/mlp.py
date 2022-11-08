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
        forward_out = np.dot(X, self.params["W"]) + self.params["b"]
        return forward_out

    def backward(self, X, grad):
        self.gradient["W"] = np.matmul(X.T, grad)
        self.gradient["b"] = np.array([np.sum(grad, axis=0)])
        backward_out = np.matmul(grad, self.params["W"].T)
        return backward_out


class ReLU:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, X):
        forward_out = np.maximum(0, X)
        return forward_out

    def backward(self, X, grad):
        new_X = np.where(X <= 0, 0, 1)
        backward_out = np.multiply(grad, new_X)
        return backward_out


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
    test_set = DataBatchMaker(X_val, y_val)

    # Momentum
    if alpha > 0.0:
        momentum = add_momentum(model)
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []

    train_loss_record = []
    val_loss_record = []

    for n in range(n_epoch):
        print("At epoch " + str(n + 1))
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
            x, y = train_set.get(idx_order[i * mini_batch_size : (i + 1) * mini_batch_size])

            # Forward Pass
            a1 = model["FC_1"].forward(x)
            h1 = model["RELU_1"].forward(a1)
            a2 = model["FC_2"].forward(h1)
            h2 = model["RELU_2"].forward(a2)
            a3 = model["FC_3"].forward(h2)
            h3 = model["RELU_3"].forward(a3)
            a4 = model["FC_4"].forward(h3)
            loss = model["LOSS"].forward(a4, y)

            # Backward Pass
            grad_a4 = model["LOSS"].backward(a4, y)
            grad_h3 = model["FC_4"].backward(h3, grad_a4)
            grad_a3 = model["RELU_3"].backward(a3, grad_h3)
            grad_h2 = model["FC_3"].backward(h2, grad_a3)
            grad_a2 = model["RELU_2"].backward(a2, grad_h2)
            grad_h1 = model["FC_2"].backward(h1, grad_a2)
            grad_a1 = model["RELU_1"].backward(a1, grad_h1)
            grad_x = model["FC_1"].backward(x, grad_a1)

            # Update gradient
            model = gradient_descent(model, momentum, alpha, learning_rate)

        ### Computing training accuracy and obj ###
        for i in range(int(np.floor(n_train / mini_batch_size))):

            x, y = train_set.get(np.arange(i * mini_batch_size, (i + 1) * mini_batch_size))

            ### forward pass ###
            a1 = model["FC_1"].forward(x)
            h1 = model["RELU_1"].forward(a1)
            a2 = model["FC_2"].forward(h1)
            h2 = model["RELU_2"].forward(a2)
            a3 = model["FC_3"].forward(h2)
            h3 = model["RELU_3"].forward(a3)
            a4 = model["FC_4"].forward(h3)
            loss = model["LOSS"].forward(a4, y)
            train_loss += loss
            train_acc += np.sum(predict_label(a2) == y)
            train_count += len(y)

        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        print("Training loss at epoch " + str(n + 1) + " is " + str(train_loss))
        print("Training accuracy at epoch " + str(n + 1) + " is " + str(train_acc))

        ### Computing validation accuracy ###
        for i in range(int(np.floor(n_val / mini_batch_size))):

            x, y = test_set.get(np.arange(i * mini_batch_size, (i + 1) * mini_batch_size))

            ### forward pass ###
            a1 = model["FC_1"].forward(x)
            h1 = model["RELU_1"].forward(a1)
            a2 = model["FC_2"].forward(h1)
            h2 = model["RELU_2"].forward(a2)
            a3 = model["FC_3"].forward(h2)
            h3 = model["RELU_3"].forward(a3)
            a4 = model["FC_4"].forward(h3)
            loss = model["LOSS"].forward(a4, y)
            val_loss += loss
            val_acc += np.sum(predict_label(a4) == y)
            val_count += len(y)

        val_loss_record.append(val_loss)
        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)

        print("Validation accuracy at epoch " + str(n + 1) + " is " + str(val_acc))

    return model


def predict(model, X):
    a1 = model["FC_1"].forward(X)
    h1 = model["RELU_1"].forward(a1)
    a2 = model["FC_2"].forward(h1)
    h2 = model["RELU_2"].forward(a2)
    a3 = model["FC_3"].forward(h2)
    h3 = model["RELU_3"].forward(a3)
    a4 = model["FC_4"].forward(h3)
    labels = predict_label(a4)
    return labels
