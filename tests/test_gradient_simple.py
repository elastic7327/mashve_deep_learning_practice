#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

def func_2(x):
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h

        fxh1 = f(x)

        x[idx] = tmp_val - h

        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

@pytest.mark.skip(reason="skip it for a moment")
def test_numerical_gradient():
    res = numerical_gradient(func_2, np.array([3.0, 4.0]))
    print(res)

    res = numerical_gradient(func_2, np.array([0.0, 2.0]))
    print(res)

    res = numerical_gradient(func_2, np.array([3.0, 0.0]))
    print(res)

def test_gradient_descent():
    init_x = np.array([-3.0, 4.0])
    res = gradient_descent(func_2, init_x=init_x, step_num=100)
    print(res)
    pass

