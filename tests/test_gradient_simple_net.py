#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from deep_learning_from_scratch.gradient_simplenet import simpleNet
from deep_learning_from_scratch.gradient import numerical_gradient


def test_simple_net_class():
    net = simpleNet()

    print(net.W)
    print(net.W.shape)
    print(net.W.ndim)

    x = np.array([0.6, 0.9])

    print(x)
    print(x.shape)
    print(x.ndim)


    p = net.predict(x)

    print(p)
    print(p.ndim)
    print(p.shape)

    arg_idx = np.argmax(p) # 최댓값의 인덱스

    t = np.array([0, 0, 1]) # 정답레이블

    print (net.loss(x, t))

    def f(W):
        return net.loss(x, t)

    dW = numerical_gradient(f, net.W)

    __import__('ipdb').set_trace()
