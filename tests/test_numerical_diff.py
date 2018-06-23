
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_2(x):
    return x[0]**2 + x[1]**2

def function_1(x):
    return 0.01*x**2 + 0.1*x

@pytest.mark.skip(reason="skip it for a moment")
def test_sample_function_draw():
    x = np.arange(0.0, 20.0, 0.1) # 0 ~ 20 0.1 단위로 나눠서
    y = function_1(x)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()

def test_numerical_diff_draw():
    # res1 = numerical_diff(function_1, 5)
    # res2 = numerical_diff(function_1, 10)

    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    g1 = numerical_diff(function_1, x)

    plt.plot(x, y, label="func")
    plt.plot(x, g1, linestyle="--", label="diff")

    plt.show()

    # print(res1)
    # print(res2)


@pytest.mark.skip(reason="skip it for a moment")
def test_logx_graph():

    x = np.arange(0, 1, 0.01)
    y = np.log(x)

    plt.xlabel("x")
    plt.xlabel("y")
    plt.plot(x, y)
    plt.show()

