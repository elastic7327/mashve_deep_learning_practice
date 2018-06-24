#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from deep_learning_from_scratch.functions import softmax, cross_entropy_error
from deep_learning_from_scratch.gradient import numerical_gradient

class simpleNet:

    """Docstring for simpleNet. """

    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

