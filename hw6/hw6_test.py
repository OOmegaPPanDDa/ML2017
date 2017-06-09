# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:50:19 2017

@author: HSIN
"""

import csv
import pandas
import numpy as np
import sys

import keras.backend as K
import keras
from keras.layers import Input, Embedding, Flatten, Add, Dot
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint





n_movies = 3953
n_items = n_movies

n_users = 6041



    

tests = pandas.read_csv(sys.argv[1], sep=',', engine='python', header = 0,
                          names=['testid', 'userid', 'movieid']).set_index('testid')
                          
                          
tests.movieid = tests.movieid.astype('category')
tests.userid = tests.userid.astype('category')

test_id = tests.index.values
tests_movieid = np.asarray(tests.movieid.values, dtype='int')
tests_userid = np.asarray(tests.userid.values, dtype='int')







def rmse_score(y_true,y_pred):
      
    return K.mean((y_true - y_pred) ** 2) ** 0.5
    


def create_model(n_users, n_items, latent_dim=20):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = keras.models.Model([user_input, item_input], r_hat)
    
    
    return model





model = create_model(n_users, n_items)
model.compile(loss='mse', optimizer='adam', metrics=[rmse_score])
model.summary()


batch_size = 512



model = load_model('mf_complete_model.h5')



res = model.predict([tests_userid,tests_movieid])


pred = res.flatten()


result = [['TestDataID','Rating']]
for i, j in zip(test_id, pred):
    line = []
    line.append(int(i))
    line.append(float(j))
    result.append(line)

f = open(sys.argv[2], 'w', encoding = 'big5', newline='')
w = csv.writer(f)
w.writerows(result)
f.close()
