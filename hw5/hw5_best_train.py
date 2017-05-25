# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:27:15 2017

@author: HSIN
"""

import sys
import csv
import pickle
import string
import numpy as np


from sklearn.preprocessing import MultiLabelBinarizer

import keras.backend as K 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model
from keras.models import Sequential
from keras.models import Model

from keras.layers import Input, Dense,Dropout
from keras.layers import GRU
from keras.layers import merge

from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.advanced_activations import PReLU
from keras.layers.embeddings import Embedding

from keras.callbacks import EarlyStopping, ModelCheckpoint






"""
nltk.download('stopwords')
stop = set(stopwords.words('english'))
"""


stop = {'through', 'him', 'were', 'more', 'when', 'was', 'such', 'your', 
'any', 'how', 'on', 'am', 'at', 'having', 'into', 'by', 'is', 'd', 'whom', 'about', 
'their', 'few', 'mustn', 'only', 'his', 'who', 'have', 'wasn', 'once', 'over', 
'here', 'for', 'other', 'nor', 'again', 'from', 'theirs', 'being', 'in', 'these', 
'some', 'or', 'same', 'had', 's', 'there', 'because', 'can', 'needn', 'what', 
'out', 'be', 'our', 'yourself', 'will', 'he', 'both', 'each', 'of', 'then', 'just', 
'herself', 'own', 'been', 'an', 'me', 'above', 'himself', 'those', 'now', 'isn', 
'them', 'doing', 'so', 'all', 'ourselves', 'do', 'off', 'while', 'are', 'don', 'up', 
'ours', 'doesn', 'a', 'until', 'you', 'most', 'they', 'under', 'my', 'this', 
'no', 'm', 'she', 'as', 'between', 'hasn', 'did', 'shan', 'weren', 'very', 're', 
'hers', 'o', 'why', 'but', 'didn', 'we', 'which', 'i', 'than', 'to', 'themselves', 
'after', 'and', 'itself', 'll', 'if', 'has', 'hadn', 'not', 'mightn', 'couldn', 
'does', 'should', 'her', 'myself', 'further', 've', 'before', 'it', 'shouldn', 
'against', 'below', 'ain', 'yours', 'y', 'with', 'during', 'too', 'wouldn', 'haven', 
'ma', 'won', 't', 'that', 'yourselves', 'its', 'where', 'aren', 'down', 'the'}




translator = str.maketrans('', '', string.punctuation)

thresh = 0.4
bag_voter_num = 10
bag_patience = 5
bag_epochs = 1000

rnn_voter_num = 10
rnn_patience = 25
rnn_epochs = 1000

batch_size = 128


text_len = []


res_box = []
model_f1_scores = []


train_texts = []
train_labels = []
# f = open('train_data.csv','r')
f = open(sys.argv[1],'r')
train_header = f.readline()
for row in f:
    index = row[:row.find(',')]
    rest_row = row[row.find(',')+1:]
    label = set(rest_row[1:rest_row.find(',')-1].split(' '))
    rest_row = rest_row[rest_row.find(',')+1:]
    
    # text = rest_row
    
    text = rest_row.translate(translator)
    text = [i for i in text.lower().split() if i not in stop]
    text_len.append(len(text))
    text = ' '.join(text)
    
    train_texts.append(text)
    train_labels.append(label)


test_texts = []

# f = open('test_data.csv','r')
f = open(sys.argv[2])
test_header = f.readline()
for row in f:
    index = row[:row.find(',')]
    rest_row = row[row.find(',')+1:]
    

    # text = rest_row
    
    text = rest_row.translate(translator)
    text = [i for i in text.lower().split() if i not in stop]
    text_len.append(len(text))
    text = ' '.join(text)
 
    test_texts.append(text)
    
f.close()

all_texts = train_texts + test_texts




print('text_len max: ', np.max(text_len))
print('text_len min: ', np.min(text_len))
print('text_len mean: ', np.mean(text_len))
print('text_len median: ', np.median(text_len))


"""
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)

pickle.dump(tokenizer, open('tokenizer', 'wb'))

del tokenizer
"""


tokenizer = pickle.load(open('tokenizer', 'rb'))


word_index = tokenizer.word_index
num_words = len(word_index) + 1

print(len(tokenizer.word_index))


#################################
# Bag
#################################

bag_mode = 'tfidf'

X_train = tokenizer.texts_to_matrix(train_texts, mode=bag_mode)
X_test = tokenizer.texts_to_matrix(test_texts, mode=bag_mode)


mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)

with open('mlb','wb') as fp:
    pickle.dump(mlb, fp)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')

print(list(mlb.classes_))



def precision(y_true,y_pred):
    
    # if (K.sum(y_pred > thresh) == 0):
    #     y_pred[np.argmax(y_pred)] = 1


    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred, axis = 1)
    
    precision=tp/((K.sum(y_pred, axis = 1))+K.epsilon())
    # recall=tp/((K.sum(y_true, axis = 1))+K.epsilon())
    return (K.mean(precision))
    # return (K.mean(recall))
    # return (K.mean(2*((precision*recall)/((precision+recall)+K.epsilon()))))



def recall(y_true,y_pred):

    # if (K.sum(y_pred > thresh) == 0):
    #     y_pred[np.argmax(y_pred)] = 1


    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred, axis = 1)

    # precision=tp/((K.sum(y_pred, axis = 1))+K.epsilon())
    recall=tp/((K.sum(y_true, axis = 1))+K.epsilon())
    # return (K.mean(precision))
    return (K.mean(recall))
    # return (K.mean(2*((precision*recall)/((precision+recall)+K.epsilon()))))


def f1_score(y_true,y_pred):

    # if (K.sum(y_pred > thresh) == 0):
    #     y_pred[np.argmax(y_pred)] = 1
    


    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred, axis = 1)

    precision=tp/((K.sum(y_pred, axis = 1))+K.epsilon())
    recall=tp/((K.sum(y_true, axis = 1))+K.epsilon())

    # return (K.mean(precision))
    # return (K.mean(recall))
    return (K.mean(2*((precision*recall)/((precision+recall)+K.epsilon()))))





def create_bag_model(selection_num):
        
    if (selection_num == 0):
        model = Sequential()
    
        model.add(Dense(1024, input_dim = X_train.shape[1]))
        model.add(PReLU())
        model.add(Dropout(drop_out_ratio))
        
        model.add(Dense(512))
        model.add(PReLU())
        model.add(Dropout(drop_out_ratio))  


        model.add(Dense(256))
        model.add(PReLU())
        model.add(Dropout(drop_out_ratio))
        
        model.add(Dense(128))
        model.add(PReLU())
        model.add(Dropout(drop_out_ratio))
        

        model.add(Dense(y_train.shape[1],activation='sigmoid'))
        

    return model







        

for voter in range(bag_voter_num):
    
    print('bag voter number: ', voter)

    np.random.seed(seed = 5246 + 46 * voter)

    train_valid_ratio = 0.9
    indices = np.random.permutation(X_train.shape[0])
    train_idx, valid_idx = indices[:int(X_train.shape[0] * train_valid_ratio)], indices[int(X_train.shape[0] * train_valid_ratio):]
    the_X_train, the_X_valid = X_train[train_idx,:], X_train[valid_idx,:]
    the_y_train, the_y_valid = y_train[train_idx,:], y_train[valid_idx,:]


    drop_out_ratio = 0.3

    epochs = bag_epochs
    patience = bag_patience


    model = create_bag_model(voter % 1)
    model.summary()



    # rdam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[precision, recall, f1_score])

    earlystopping = EarlyStopping(monitor='val_f1_score', patience = patience, verbose=1, mode='max')


    checkpoint = ModelCheckpoint('best_bag_model'+str(voter)+'.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')

    history = model.fit(the_X_train, the_y_train, 
                     epochs = epochs, 
                     batch_size = batch_size,
                     validation_data = (the_X_valid, the_y_valid),
                     callbacks=[earlystopping,checkpoint])



    del model


    model = create_bag_model(voter % 1)
    model.load_weights('best_bag_model'+str(voter)+'.hdf5')

    model.save('best_bag_complete_model'+str(voter)+'.h5')
    del model


    model = load_model('best_bag_complete_model'+str(voter)+'.h5')

    pred = model.predict(X_test, batch_size=batch_size)

    res = []

    for line in pred:
        
        bin_record = line
        
        
        if(sum(line > thresh) == 0):
            bin_record[np.argmax(line)] = 1
            bin_record[line <= thresh] = 0
        
        else:
            bin_record[line > thresh] = 1
            bin_record[line <= thresh] = 0
        
        
        
        res.append(bin_record)

    res_box.append(res)

    print('model f1 score: ', np.max(history.history['val_f1_score']))
    model_f1_scores.append(np.max(history.history['val_f1_score']))
    print(model_f1_scores)





#################################
# RNN
#################################
train_sequences = tokenizer.texts_to_sequences(train_texts)
X_train = pad_sequences(train_sequences)
maxlen = X_train.shape[1]

# maxlen = 186
print(maxlen)

X_train = pad_sequences(train_sequences, maxlen=maxlen)

test_sequences = tokenizer.texts_to_sequences(test_texts)
X_test = pad_sequences(test_sequences, maxlen=maxlen)


"""
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)
"""

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')

print(list(mlb.classes_))



embedding_dict = {}
word_vec_dim = 100




# glove method
f = open('glove.6B.%dd.txt' % word_vec_dim)
# f = open('glove.840B.%dd.txt' % word_vec_dim)
for line in f:
    values = line.split(' ')
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    embedding_dict[word] = vec
f.close()
    



embedding_matrix = np.zeros((num_words, word_vec_dim))
for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector



def create_rnn_model(selection_num):
        
    if (selection_num == 0):
        model = Sequential()
        model.add(Embedding(num_words,
                            word_vec_dim,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False))
        
        
        
        model.add((GRU(256, dropout=drop_out_ratio, recurrent_dropout=drop_out_ratio, return_sequences = True)))
        model.add((GRU(128, dropout=drop_out_ratio, recurrent_dropout=drop_out_ratio)))


        model.add(Dense(64,activation='relu'))
        model.add(Dropout(drop_out_ratio))
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(drop_out_ratio))
         


        model.add(Dense(y_train.shape[1],activation='sigmoid'))
        


    if (selection_num == 1):   
        main_input = Input(shape=(maxlen, ), dtype='int32', name='main_input')

        embedding  = Embedding(num_words,
                            word_vec_dim,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=False)(main_input)

        embedding = Dropout(drop_out_ratio)(embedding)

        conv4 = Conv1D(nb_filter=200,
                              filter_length=4,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1,
                              name='conv4')(embedding)
        maxConv4 = MaxPooling1D(pool_length=2,
                                 name='maxConv4')(conv4)

        conv5 = Conv1D(nb_filter=200,
                              filter_length=5,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1,
                              name='conv5')(embedding)
        maxConv5 = MaxPooling1D(pool_length=2,
                                name='maxConv5')(conv5)

        x = merge([maxConv4, maxConv5], mode='concat')

        x = Dropout(drop_out_ratio)(x)

        x = merge([maxConv4, maxConv5], mode='concat')

        x = Dropout(drop_out_ratio)(x)

        x = (GRU(256, dropout=drop_out_ratio, recurrent_dropout=drop_out_ratio, return_sequences = True))(x)
        x = (GRU(128, dropout=drop_out_ratio, recurrent_dropout=drop_out_ratio))(x)

        x = Dense(64)(x)

        x = Dropout(drop_out_ratio)(x)

        x = Dense(64)(x)

        x = Dropout(drop_out_ratio)(x)

        output = Dense(y_train.shape[1], activation='sigmoid', name='output')(x)

        model = Model(input=main_input, output=output)
        

    return model



for voter in range(rnn_voter_num):
    
    print('rnn voter number: ', voter)

    np.random.seed(seed = 1746 + 46 * voter)

    train_valid_ratio = 0.9
    indices = np.random.permutation(X_train.shape[0])
    train_idx, valid_idx = indices[:int(X_train.shape[0] * train_valid_ratio)], indices[int(X_train.shape[0] * train_valid_ratio):]
    the_X_train, the_X_valid = X_train[train_idx,:], X_train[valid_idx,:]
    the_y_train, the_y_valid = y_train[train_idx,:], y_train[valid_idx,:]


    drop_out_ratio = 0.3

    epochs = rnn_epochs
    patience = rnn_patience


    model = create_rnn_model(voter % 2)
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[precision, recall, f1_score])

    earlystopping = EarlyStopping(monitor='val_f1_score', patience = patience, verbose=1, mode='max')


    checkpoint = ModelCheckpoint('best_rnn_model'+str(voter)+'.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')

    history = model.fit(the_X_train, the_y_train, 
                     epochs = epochs, 
                     batch_size = batch_size,
                     validation_data = (the_X_valid, the_y_valid),
                     callbacks=[earlystopping,checkpoint])



    del model


    model = create_rnn_model(voter % 2)
    model.load_weights('best_rnn_model'+str(voter)+'.hdf5')

    model.save('best_rnn_complete_model'+str(voter)+'.h5')
    del model


    model = load_model('best_rnn_complete_model'+str(voter)+'.h5')

    pred = model.predict(X_test, batch_size=batch_size)

    res = []

    for line in pred:
        
        bin_record = line
        
        
        if(sum(line > thresh) == 0):
            bin_record[np.argmax(line)] = 1
            bin_record[line <= thresh] = 0
        
        else:
            bin_record[line > thresh] = 1
            bin_record[line <= thresh] = 0
        
        res.append(bin_record)

    res_box.append(res)

    print('model f1 score: ', np.max(history.history['val_f1_score']))
    model_f1_scores.append(np.max(history.history['val_f1_score']))
    print(model_f1_scores)


    

"""

res_box = np.asarray(res_box)
print('res_box_shape', res_box.shape)
print('model_f1_scores')
print(model_f1_scores)

res_vote = np.sum(res_box, axis=0)/res_box.shape[0]



res = []

for line in res_vote:
        
    record = line
    
    record[line >= 0.5] = 1
    record[line < 0.5] = 0
    
    res.append(record)

res = np.asarray(res)
res = mlb.inverse_transform(res)




result = [["id","tags"]]
for i, j in enumerate(res,0):
    line = []
    line.append(str(i))
    line.append(str(' '.join(j)))
    result.append(line)
    
    
f = open('prediction.csv', 'w', encoding = 'big5')
w = csv.writer(f,  quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
w.writerows(result)
f.close()

"""