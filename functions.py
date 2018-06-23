#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error_for_multiple_dim(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.type)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

