# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:50:13 2019

@author: chenc
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
np.set_printoptions(precision=3, suppress=True)

X = np.array([[-0.1, 1.4],
              [-0.5, 0.2],
	      [ 1.3, 0.9],
	      [-0.6, 0.4],
	      [-1.6, 0.2],
	      [ 0.2, 0.2],
	      [-0.3,-0.4],
	      [ 0.7,-0.8],
	      [ 1.1,-1.5],
	      [-1.0, 0.9],
	      [-0.5, 1.5],
	      [-1.3,-0.4],
	      [-1.4,-1.2],
	      [-0.9,-0.7],
	      [ 0.4,-1.3],
	      [-0.4, 0.6],
	      [ 0.3,-0.5],
	      [-1.6,-0.7],
	      [-0.5,-1.4],
	      [-1.0,-1.4]])

y = np.array([0, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 2, 2, 2, 1, 0, 1, 2, 2, 2])
Y = np.eye(3)[y]
