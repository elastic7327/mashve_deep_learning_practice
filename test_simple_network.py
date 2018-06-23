#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import matplotlib.pyplot as plt

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def advanced_step_function(x):
    return np.array(x > 0, dtype=np.int )

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def relu(x):
    return np.maximum(0, x)

@pytest.mark.skip(reason="skip it for a moment")
def test_step_function():
    assert step_function(1) == 1
    assert step_function(0) == 0

@pytest.mark.skip(reason="skip it for a moment")
def test_advanced_step_func():
    advanced_step_function(np.array([10, 20, 30, 1, 2, 0]))

@pytest.mark.skip(reason="skip it for a moment")
def test_ad_step_func_drawing():
    x = np.arange(-5.0, 5.0, 0.1)
    y = advanced_step_function(x)

    plt.plot(x, y)
    # plt.ylim(-0.1, 1.1) # y축 범위지정
    plt.show()

@pytest.mark.skip(reason="skip it for a moment")
def test_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)

    plt.plot(x, y)
    # plt.ylim(-0.1, 1.1) # y축 범위지정
    plt.show()

@pytest.mark.skip(reason="skip it for a moment")
def test_RelU():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)

    plt.plot(x, y)
    # plt.ylim(-0.1, 1.1) # y축 범위지정
    plt.show()
