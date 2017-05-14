# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:17:00 2017

@author: HSIN
"""


import numpy as np
import csv
import sys
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors




def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


def get_transform(data):
    randidx = np.random.permutation(data.shape[0])[:3]
    knbrs = NearestNeighbors(n_neighbors=200, algorithm='ball_tree').fit(data)
    

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors([data[idx]])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals



np.random.seed(seed = 46)



X = []
y = []
for i in range(60):
    dim = i + 1
    # print(dim)
    for j in range(10):
        N = np.random.randint(10000,100000)
        layer_dims = [np.random.randint(60, 79), 100]
        data = gen_data(dim, layer_dims, N).astype('float32')
        the_data = get_transform(data)
        X.append(the_data)
        y.append(np.log(dim))



X = np.array(X)
y = np.array(y)

np.savez('train_data.npz', X=X, y=y)

npzfile = np.load('train_data.npz')
X = npzfile['X']
y = npzfile['y']


svr = SVR(C=1)
svr.fit(X, y)


# predict

#testdata = np.load('data.npz')
testdata = np.load(sys.argv[1])

test_X = []
for i in range(200):
    data = testdata[str(i)]
    the_data = get_transform(data)
    test_X.append(the_data)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)



result = [['SetId','LogDim']]
for i, j in enumerate(pred_y):
  line = []
  line.append(i)
  line.append(j)
  result.append(line)


#f = open('prediction.csv', 'w')
f = open(sys.argv[2], 'w')
w = csv.writer(f)
w.writerows(result)
f.close()