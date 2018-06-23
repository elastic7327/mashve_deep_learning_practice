#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )
