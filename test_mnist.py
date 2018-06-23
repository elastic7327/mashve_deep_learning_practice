#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import sys, os
import pytest
import numpy as np

from softmax import softmax

from test_simple_network import sigmoid

sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from PIL import Image

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test

def init_network():
    with open("dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error_for_multiple_dim(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.type)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3

    y = softmax(a3)
    return y

@pytest.mark.skip(reason="skip it for a moment")
def test_simple_mnist_image_print():
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    img = x_train[0]
    label = y_train[0]

    print(label) # 5
    print(img.shape)

    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)

@pytest.mark.skip(reason="skip it for a moment")
def test_simple_mnist():
    x, _ = get_data()
    network = init_network()

    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    print(x.shape)
    print(x[0].shape)
    print(W1.shape)
    print(W2.shape)
    print(W3.shape)


@pytest.mark.skip(reason="skip it for a moment")
def test_mnist_with_batch():
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt)/ len(x)))

@pytest.mark.skip(reason="skip it for a moment")
def test_mean_squared_error():
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # 2일 확률이 가장 높다고 추정함
    # y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

    # 7일 확률이 가장 높다고 추정함
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

    res = mean_squared_error(np.array(y), np.array(t))
    print(res)


@pytest.mark.skip(reason="skip it for a moment")
def test_cross_entropy_error():
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

    res = cross_entropy_error(np.array(y), np.array(t))
    print(res)


def test_mini_batch():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    print(x_train.shape)
    print(t_train.shape)

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size) # 가령 10000, 10 이라고하면 10000미만0이상 값중 10개를 뽑아냅니다.

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
