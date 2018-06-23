#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def identity(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([0.1, 0.3, 0.4])

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

@pytest.mark.skip(reason="skip it for a moment")
def test_multi_array():
    A = np.array([1, 2, 3, 4, 5])
    print(A)
    print("Number of Dimentions", np.ndim(A))
    print("Shapes", A.shape)
    print(A.shape[0])

    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(B)
    print("Number of Dimentions", np.ndim(B))
    print("Shapes", B.shape)
    print(B.shape[0])

    C = np.array([[1, 2], [3, 4]])
    D = np.array([[3, 4], [5, 6]])

    print(np.dot(C, D))

@pytest.mark.skip(reason="skip it for a moment")
def test_matrix_dot():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    print(A.shape)
    B = np.array([7, 8])
    print(np.dot(A, B))

@pytest.mark.skip(reason="skip it for a moment")
def test_simple_net_multi():
    X = np.array([1, 2], dtype=int)
    W = np.array([[1, 3, 5], [2, 4, 6]], dtype=int)
    print(np.dot(X, W))

@pytest.mark.skip(reason="skip it for a moment")
def test_simple_net_multi2():

    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 3 x 2
    B1 = np.array([0.1, 0.2, 0.3])

    print(W1.shape)
    print(X.shape)
    print(B1.shape)

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)

    print(Z1) # 3 x 1

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 2 x 3
    B2 = np.array([0.1, 0.2])

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print(Z2)

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    A3 = np.dot(Z2, W3) + B3
    print(A3)
    Y = identity(A3)

@pytest.mark.skip(reason="skip it for a moment")
def test_simple_net_summary():
    a = np.array([1010, 1000, 990])
    # softmax = np.exp(a) / np.sum(np.exp(a))
    c = np.max(a)
    res = np.exp(a - c ) / np.sum(np.exp(a - c))

def test_softmax_func():
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
    res = np.sum(y)
    print(res)

