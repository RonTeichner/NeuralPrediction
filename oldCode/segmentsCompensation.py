#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:51:41 2023

@author: ron.teichner
"""

import numpy as np
import matplotlib.pyplot as plt


d01 = 10
d12 = 10
d23 = 10

x0 = np.random.randn(1000)
x1 = x0 + d01 + np.random.randn(1000)
x2 = x1 + d12 + np.random.randn(1000)
x3 = x2 + d23 + np.random.randn(1000)

dx2 = x2-x1
dx3 = x3-x2

plt.figure()
plt.scatter(x=dx2, y=dx3, s=10)
plt.xlabel('segment n-1')
plt.ylabel('segment n')
plt.gird()
plt.show()