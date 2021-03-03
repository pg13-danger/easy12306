# coding: utf-8
import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


def to_categorical(y, num_classes):
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype='float32')
    categorical[np.arange(n), y] = 1
    return categorical


def load_data(fn='texts.npz', to=False):
    data = np.load(fn)
    texts, labels = data['texts'], data['labels']
    texts = texts.astype('float32')
    texts /= 255.0
    labels = labels.astype('int64')
    n, h, w = texts.shape
    texts.shape = (-1, 1, h, w)
    if to:
        labels = to_categorical(labels, 80)
    n = int(n * 0.9)    # 90%用于训练，10%用于测试
    return (texts[:n], labels[:n]), (texts[n:], labels[n:])


def load_data_v2():
    (train_x, train_y), (test_x, test_y) = load_data(to=True)
    # 这里是统计学数据
    (train_v2_x, train_v2_y), (test_v2_x, test_v2_y) = load_data('texts.v2.npz')
    # 合并
    train_x = np.concatenate((train_x, train_v2_x))
    train_y = np.concatenate((train_y, train_v2_y))
    test_x = np.concatenate((test_x, test_v2_x))
    test_y = np.concatenate((test_y, test_v2_y))
    return (train_x, train_y), (test_x, test_y)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 80)

    def forward(self, x, training=True):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.avg_pool2d(x, (2, 7))
        x = x.view(-1, 64)
        x = F.dropout(x, 0.25, training=training)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = self.forward(x, training=False)
        return F.softmax(x, dim=-1)


def gen_numpy_loader(x, y, shuffle=True):
    device = torch.device('cuda')
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=shuffle)


def main_v10():
    device = torch.device("cuda")
    model = Model().to(device)

    dtype = torch.float
    (train_x, train_y), (test_x, test_y) = load_data()

    train_loader = gen_numpy_loader(train_x, train_y, shuffle=True)
    test_loader = gen_numpy_loader(test_x, test_y)

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    for i in range(100):
        m = 0
        _loss = 0
        for x, y in train_loader:
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            y_pred = F.log_softmax(y_pred, dim=1)
            loss = F.nll_loss(y_pred, y)
            _loss += loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m += 1

        with torch.no_grad():
            n = 0
            val_loss = 0
            for x, y in test_loader:
                y_pred = model.forward(x, training=False)
                y_pred = F.log_softmax(y_pred, dim=1)
                val_loss += F.nll_loss(y_pred, y)
                n += 1
        scheduler.step(val_loss)
        print(i, 'loss:', _loss / m, 'val_loss:', val_loss.item() / n)

    torch.save(model.state_dict(), 'parameter.pkl')


# https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_hinge
def categorical_hinge(y_true, y_pred):
    y_pred = F.softmax(y_pred, dim=1)
    neg, _ = torch.max((1 - y_true) * y_pred, dim=1)
    pos = torch.sum(y_true * y_pred, dim=1)
    loss = neg - pos + 1
    return torch.sum(loss)


def main_v19():
    model = Model()
    model.load_state_dict(torch.load('parameter.pkl'))

    device = torch.device("cuda")
    model = model.to(device)

    dtype = torch.float
    (train_x, train_y), (test_x, test_y) = load_data_v2()

    train_loader = gen_numpy_loader(train_x, train_y, shuffle=True)
    test_loader = gen_numpy_loader(test_x, test_y)

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)
    for i in range(100):
        m = 0
        _loss = 0
        for x, y in train_loader:
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Compute and print loss
            loss = categorical_hinge(y, y_pred)
            _loss += loss.item()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m += 1

        with torch.no_grad():
            n = 0
            val_loss = 0
            for x, y in test_loader:
                y_pred = model.forward(x, training=False)
                val_loss += categorical_hinge(y, y_pred)
                n += 1
        scheduler.step(val_loss)
        print(i, 'loss:', _loss / m, 'val_loss:', val_loss.item() / n)

    torch.save(model.state_dict(), 'parameter.v19.pkl')


def acc(y_true, y_pred):
    s = np.argmax(y_true + y_pred, axis=-1) == np.argmax(y_pred, axis=-1)
    return s.mean()


def predict(path='parameter.pkl'):
    model = Model()
    model.load_state_dict(torch.load(path))
    _, (test_x, test_y) = load_data_v2()
    x = test_x
    y_prev = model.predict(x)
    y_prev = y_prev.detach().numpy()
    cv2.imwrite('a.jpg', test_x[-1, 0] * 255)
    print(y_prev[-1])
    print(y_prev[-1].argmax())
    print(acc(test_y, y_prev))


if __name__ == '__main__':
    # main_v10()
    # main_v19()
    predict('parameter.v19.pkl')
