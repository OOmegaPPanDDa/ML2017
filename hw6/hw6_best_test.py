# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:46:16 2017

@author: HSIN
"""
import csv
import pandas
import numpy as np
import sys


import keras.backend as K
import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


embed_dim = 64


users = pandas.read_csv(sys.argv[1], sep='::', engine='python', header = 0,
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')
                        
movies = pandas.read_csv(sys.argv[2], sep='::', engine='python', header = 0,
                         names=['movieid', 'title', 'genre']).set_index('movieid')                        
                        



movies['genre'] = movies.genre.str.split('|')

movies_genres_names = ['Action', 'Adventure', 'Animation', "Children's",
'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']



users.age = users.age.astype('category')
users.gender = users.gender.astype('category')
users.occupation = users.occupation.astype('category')



#n_movies = np.max(movieid)+1
#n_users = np.max(userid)+1


n_movies = 3953
n_users = 6041

    

tests = pandas.read_csv(sys.argv[3], sep=',', engine='python', header = 0,
                          names=['testid', 'userid', 'movieid']).set_index('testid')
                          
                          
tests.movieid = tests.movieid.astype('category')
tests.userid = tests.userid.astype('category')

test_id = tests.index.values
tests_movieid = np.asarray(tests.movieid.values, dtype='int16')
tests_userid = np.asarray(tests.userid.values, dtype='int16')


tests_moviegenre = movies.loc[tests_movieid].genre

tests_mlb_moviegenre = []

for genres in tests_moviegenre:
    mlb_genre = np.zeros(embed_dim)
    for g in genres:
        mlb_genre[movies_genres_names.index(g)] = 1
    tests_mlb_moviegenre.append(mlb_genre)
    
tests_mlb_moviegenre = np.asarray(tests_mlb_moviegenre)



tests_userage = np.asarray(users.loc[tests_userid].age.replace(1,1).replace(18,2).replace(25,3).replace(35,4).replace(45,5).replace(50,6).replace(56,7))

temp = []
for x in tests_userage:
    vec = np.zeros(embed_dim)
    vec[x] = 1
    temp.append(vec)
tests_userage = np.asarray(temp)




tests_usergender = np.asarray(users.loc[tests_userid].gender.replace('M',1).replace('F',0))



temp = []
for x in tests_usergender:
    vec = np.zeros(embed_dim)
    vec[x] = 1
    temp.append(vec)
tests_usergender = np.asarray(temp)




tests_userocc = np.asarray(users.loc[tests_userid].occupation)



temp = []
for x in tests_userocc:
    vec = np.zeros(embed_dim)
    vec[x] = 1
    temp.append(vec)
tests_userocc = np.asarray(temp)




def rmse_score(y_true,y_pred):
    
    return K.mean((y_true * 5 - y_pred * 5) ** 2) ** 0.5
    


def create_model():
    movie_input = keras.layers.Input(shape=[1])
    movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, embed_dim)(movie_input))
    movie_vec = keras.layers.Dropout(0.5)(movie_vec)
    
    user_input = keras.layers.Input(shape=[1])
    user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, embed_dim)(user_input))
    user_vec = keras.layers.Dropout(0.5)(user_vec)  
    
    moviegenre_input = keras.layers.Input(shape=[embed_dim])
    moviegenre_vec = keras.layers.Dropout(0.5)(moviegenre_input)
    
    userage_input = keras.layers.Input(shape=[embed_dim])
    userage_vec = keras.layers.Dropout(0.5)(userage_input)
    
    usergender_input = keras.layers.Input(shape=[embed_dim])
    usergender_vec = keras.layers.Dropout(0.5)(usergender_input)
    
    
    userocc_input = keras.layers.Input(shape=[embed_dim])
    userocc_vec = keras.layers.Dropout(0.5)(userocc_input)
    
    input_vecs = keras.layers.add([movie_vec, user_vec, moviegenre_vec, userage_vec, usergender_vec, userocc_vec])
    
    nn = keras.layers.Dropout(0.5)(keras.layers.Dense(512, activation='relu')(input_vecs))
    nn = keras.layers.normalization.BatchNormalization()(nn)
    
    nn = keras.layers.Dropout(0.5)(keras.layers.Dense(256, activation='relu')(nn))
    nn = keras.layers.normalization.BatchNormalization()(nn)
    
    nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
    nn = keras.layers.normalization.BatchNormalization()(nn)
    
    
    nn = keras.layers.Dense(128, activation='relu')(nn)
    
    result = keras.layers.Dense(1, activation='sigmoid')(nn)
    
    model = keras.models.Model([movie_input, user_input, moviegenre_input, userage_input, usergender_input, userocc_input], result)
    
    return model





model = create_model()
model.compile('adam', 'mean_squared_error', metrics=[rmse_score])
model.summary()


np.random.seed(seed = 1746)




batch_size = 1024


model = load_model('best_complete_model.h5')


res = model.predict([tests_movieid,tests_userid, tests_mlb_moviegenre, tests_userage, tests_usergender, tests_userocc])
pred = res.flatten() * 5


result = [['TestDataID','Rating']]
for i, j in zip(test_id, pred):
    line = []
    line.append(int(i))
    line.append(float(j))
    result.append(line)

f = open(sys.argv[4], 'w', encoding = 'big5', newline='')
w = csv.writer(f)
w.writerows(result)
f.close()

