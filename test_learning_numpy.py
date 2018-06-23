
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import logging
import matplotlib.pyplot as plt


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2

    if tmp <= 0.7:
        return 0
    else:
        return 1

def AAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # AND와는 다르게 w와 b만 다르다.
    b = 0.7
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b

    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

@pytest.mark.skip(reason="skip it for a moment")
def test_basic_numpy():

    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[3, 1], [0, 8]])
    print(arr1.dtype, arr1.shape)
    print(arr1 + arr2)

def test_numpy_broadcast():

    a = np.array([[1, 2], [3, 4]])
    b = np.array([10, 20])
    # print(a * b)

    a = a.flatten()
    # print(a)

    # print(a < 2)
    # print(a[a < 3])
    # print(a[np.array([0, 1, 2])])

    logging.debug(a < 2)

@pytest.mark.skip(reason="skip it for a moment")
def test_simple_sin_graph():
    x = np.arange(0, 6, 0.1) # 0 부터 6까지 0.1 간격으로 배열생성
    print(x)
    y = np.sin(x)

    # 그래프 그리기
    plt.plot(x, y)
    plt.show()

@pytest.mark.skip(reason="skip it for a moment")
def test_cos_and_sin_graph():
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, linestyle="--", label="cos")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("sin & cos")

    plt.legend()
    plt.show()

def test_perceptron():
    assert AND(2, 2) == 1
    assert AND(0, 1) == 0

def test_simple_numpy():
    x = np.array([0, 1])
    w = np.array([0.5, 0.5])
    b = -0.7
    z = np.sum(w*x) + b

def test_AAND_NAND_function():

    assert AAND(0.5, 1) == 1
    assert AAND(1, 0) == 0
    assert AAND(0, 1) == 0
    assert AAND(1, 1) == 1

    assert NAND(0, 0) == 1
    assert NAND(1, 0) == 1
    assert NAND(0, 1) == 1
    assert NAND(1, 1) == 0

    assert OR(0, 0) == 0
    assert OR(1, 0) == 1
    assert OR(0, 1) == 1
    assert OR(1, 1) == 1

    assert XOR(0, 0) == 0
    assert XOR(1, 0) == 1
    assert XOR(0, 1) == 1
    assert XOR(1, 1) == 0

