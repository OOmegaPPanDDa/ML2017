# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:57:26 2017

@author: HSIN
"""


import numpy as np
import sys

matrixA = np.loadtxt(sys.argv[1], delimiter = ',')
matrixB = np.loadtxt(sys.argv[2], delimiter = ',')
ans_one_matrix = sorted(np.dot(matrixA, matrixB).astype(int))
np.savetxt('ans_one.txt', ans_one_matrix, fmt='%i')
