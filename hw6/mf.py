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


dim_num = 20


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




np.random.seed(seed = 4646)


train_valid_ratio = 0.9
indices = np.random.permutation(y.shape[0])
train_idx, valid_idx = indices[:int(y.shape[0] * train_valid_ratio)], indices[int(y.shape[0] * train_valid_ratio):]

movieid_train, userid_train, y_train = movieid[train_idx], userid[train_idx], y[train_idx]

movieid_valid, userid_valid, y_valid = movieid[valid_idx], userid[valid_idx], y[valid_idx]


## normalization
#mean = np.mean(y_train)
#std = np.std(y_train)
#
#y_train = y_train - mean
#y_trian = y_train/std
#
#y_valid = y_valid - mean
#y_valid = y_valid/std




def rmse_score(y_true,y_pred):
      
     return K.mean((y_true - y_pred) ** 2) ** 0.5
    
    
#    # normalization
#    true = y_true * std + mean
#    pred = y_pred * std + mean
#    return K.mean((true - pred) ** 2) ** 0.5
    


def create_model(n_users, n_items, latent_dim=dim_num):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    
    # bias
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = keras.models.Model([user_input, item_input], r_hat)
    
    
#    # no bias
#    r_hat = Dot(axes=1)([user_vec, item_vec])
#    model = keras.models.Model([user_input, item_input], r_hat)
    
    return model





model = create_model(n_users, n_items)
model.compile(loss='mse', optimizer='adam', metrics=[rmse_score])
model.summary()








patience = 10
epochs = 1000
batch_size = 512

#earlystopping = EarlyStopping(monitor='val_rmse_score', patience = patience, verbose=1, mode='auto')
#checkpoint = ModelCheckpoint('mf_model.h5',
#                             verbose=1,
#                             save_best_only=True,
#                             save_weights_only=True,
#                             monitor='val_rmse_score',
#                             mode='min')
#
#history = model.fit([userid_train, movieid_train], y_train, 
#                         epochs=epochs, 
#                         batch_size = batch_size,
#                         validation_data=([userid_valid, movieid_valid], y_valid),
#                 		 callbacks=[earlystopping,checkpoint])
#
#
#del model
#
#model = create_model(n_users, n_items)
#model.load_weights('mf_model.h5')
#
#model.save('mf_complete_model.h5')
#del model


model = load_model('mf_complete_model.h5')


#
#train_res = model.predict([userid_train, movieid_train])
#
#train_pred = train_res.flatten()
#train_true = y_train
#
#
### normalization
##train_pred = train_pred * std + mean
##train_true = train_true * std + mean
#
#train_error = np.mean((train_true - train_pred) ** 2) ** 0.5
#print('train error: ', train_error)
#
#
#valid_res = model.predict([userid_valid, movieid_valid])
#
#
#valid_pred = valid_res.flatten()
#valid_true = y_valid
#
#
### normalization
##valid_pred = valid_pred * std + mean
##valid_true = valid_true * std + mean
#
#valid_error = np.mean((valid_true - valid_pred) ** 2) ** 0.5
#print('valid error: ', valid_error)
#
#res = model.predict([tests_userid,tests_movieid])
#
#
#pred = res.flatten()
#
### normalization
##pred = pred * std + mean
#
#
#result = [['TestDataID','Rating']]
#for i, j in zip(test_id, pred):
#    line = []
#    line.append(int(i))
#    line.append(float(j))
#    result.append(line)
#
#f = open('mf_prediction.csv', 'w', encoding = 'big5', newline='')
#w = csv.writer(f)
#w.writerows(result)
#f.close()







user_emb = np.array(model.layers[2].get_weights()).squeeze()
print('user embedding shape:', user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print('movie embedding shape:', movie_emb.shape)

np.save('user_emb.npy', user_emb)
np.save('movie_emb.npy', movie_emb)



movies = pandas.read_csv('data/movies.csv', sep='::', engine='python', header = 0,
                         names=['movieid', 'title', 'genre']).set_index('movieid')                        
                        



movies['genre'] = movies.genre.str.split('|')

movies_genres_names = ['Action', 'Adventure', 'Animation', "Children's",
'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']



movie_x = movie_emb
movie_y = np.zeros(movie_emb.shape[0])
movie_y = list(movie_y)

for i in range(movie_emb.shape[0]):
    try:
        movie_y[i+1] = movies.genre[i+1]
    except:
        continue
    


x = []
y = []
for i in range(len(movie_y)):
    if movie_y[i] == 0:
        movie_y[i] = 'None'
        
        
    if any(xs in movie_y[i] for xs in ['Action','War']):
        x.append(movie_x[i])
        y.append(-1)
    elif any(xs in movie_y[i] for xs in ['Romance','Drama','Children\'s']):
        x.append(movie_x[i])
        y.append(1)
    else:
        continue
    
    


from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
x = np.array(x, dtype = np.float64)
y = np.array(y)


tsne_model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vis_data = tsne_model.fit_transform(x) 

vis_x = vis_data[:,0]
vis_y = vis_data[:,1]


cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(vis_x, vis_y, c = y, cmap = cm, edgecolors = None)    
plt.colorbar(sc)
plt.savefig('draw.png', dpi=800)
plt.show()
