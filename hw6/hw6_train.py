# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 21:50:19 2017

@author: HSIN
"""

import csv
import pandas
import numpy as np

import keras.backend as K
import keras
from keras.layers import Input, Embedding, Flatten, Add, Dot
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


ratings = pandas.read_csv('data/train.csv', sep=',', engine='python', header = 0,
                          names=['trainid', 'userid', 'movieid', 'rating']).set_index('trainid')

ratings.movieid = ratings.movieid.astype('category')
ratings.userid = ratings.userid.astype('category')



movieid = np.asarray(ratings.movieid.values, dtype='int')
userid = np.asarray(ratings.userid.values, dtype='int')




n_movies = np.max(movieid)+1
n_items = n_movies

n_users = np.max(userid)+1



    

tests = pandas.read_csv('data/test.csv', sep=',', engine='python', header = 0,
                          names=['testid', 'userid', 'movieid']).set_index('testid')
                          
                          
tests.movieid = tests.movieid.astype('category')
tests.userid = tests.userid.astype('category')

test_id = tests.index.values
tests_movieid = np.asarray(tests.movieid.values, dtype='int')
tests_userid = np.asarray(tests.userid.values, dtype='int')




y = np.zeros((ratings.shape[0], 5))
y[np.arange(ratings.shape[0]), ratings.rating - 1] = 1


y = np.dot(y, [1,2,3,4,5])




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



np.random.seed(seed = 1746)


train_valid_ratio = 0.9
indices = np.random.permutation(y.shape[0])
train_idx, valid_idx = indices[:int(y.shape[0] * train_valid_ratio)], indices[int(y.shape[0] * train_valid_ratio):]

movieid_train, userid_train, y_train = movieid[train_idx], userid[train_idx], y[train_idx]

movieid_valid, userid_valid, y_valid = movieid[valid_idx], userid[valid_idx], y[valid_idx]


patience = 10
epochs = 1000
batch_size = 512

earlystopping = EarlyStopping(monitor='val_rmse_score', patience = patience, verbose=1, mode='auto')
checkpoint = ModelCheckpoint('mf_model.h5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_rmse_score',
                             mode='min')

history = model.fit([userid_train, movieid_train], y_train, 
                         epochs=epochs, 
                         batch_size = batch_size,
                         validation_data=([userid_valid, movieid_valid], y_valid),
                 		 callbacks=[earlystopping,checkpoint])


del model

model = create_model(n_users, n_items)
model.load_weights('mf_model.h5')

model.save('mf_complete_model.h5')
del model


model = load_model('mf_complete_model.h5')


valid_res = model.predict([userid_valid, movieid_valid])


valid_pred = valid_res.flatten()
valid_true = y_valid

valid_error = np.mean((valid_true - valid_pred) ** 2) ** 0.5
print('valid error: ', valid_error)

res = model.predict([tests_userid,tests_movieid])


pred = res.flatten()


result = [['TestDataID','Rating']]
for i, j in zip(test_id, pred):
    line = []
    line.append(int(i))
    line.append(float(j))
    result.append(line)

f = open('mf_prediction.csv', 'w', encoding = 'big5', newline='')
w = csv.writer(f)
w.writerows(result)
f.close()
