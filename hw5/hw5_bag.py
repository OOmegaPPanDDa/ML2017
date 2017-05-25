# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:06:20 2017

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


import matplotlib.pyplot as plt



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

text_len = []


train_texts = []
train_labels = []
f = open('train_data.csv','r')
# f = open(sys.argv[1],'r')
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

f = open('test_data.csv','r')
# f = open(sys.argv[2])
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


bag_mode = 'tfidf'

X_train = tokenizer.texts_to_matrix(train_texts, mode=bag_mode)
X_test = tokenizer.texts_to_matrix(test_texts, mode=bag_mode)


mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')

print(list(mlb.classes_))


np.random.seed(seed = 1746)

train_valid_ratio = 0.9
indices = np.random.permutation(X_train.shape[0])
train_idx, valid_idx = indices[:int(X_train.shape[0] * train_valid_ratio)], indices[int(X_train.shape[0] * train_valid_ratio):]
X_train, X_valid = X_train[train_idx,:], X_train[valid_idx,:]
y_train, y_valid = y_train[train_idx,:], y_train[valid_idx,:]








drop_out_ratio = 0.3


def create_model():
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


thresh = 0.4

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




epochs = 1000
batch_size = 128
patience = 5


model = create_model()
model.summary()



# rdam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=[precision, recall, f1_score])

earlystopping = EarlyStopping(monitor='val_f1_score', patience = patience, verbose=1, mode='max')
checkpoint = ModelCheckpoint('bag_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_f1_score',
                             mode='max')



history = model.fit(X_train, y_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_valid, y_valid),
                 callbacks=[earlystopping,checkpoint])


loss_list = list(history.history['loss'])
val_loss_list = list(history.history['val_loss'])

f1_list = list(history.history['f1_score'])
val_f1_list = list(history.history['val_f1_score'])

precison_list = list(history.history['precision'])
val_precison_list = list(history.history['val_precision'])

recall_list = list(history.history['recall'])
val_recall_list = list(history.history['val_recall'])


history_list = [loss_list, val_loss_list, f1_list, val_f1_list, 
                precison_list, val_precison_list, recall_list, val_recall_list]


with open('bag_history','wb') as fp:
    pickle.dump(history_list, fp)

with open('bag_history','rb') as fp:
    history_list = pickle.load(fp)



del model

model = create_model()
model.load_weights('bag_model.hdf5')

model.save('bag_complete_model.h5')
del model


model = load_model('bag_complete_model.h5')


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
        

res = np.asarray(res)
res = mlb.inverse_transform(res)




result = [["id","tags"]]
for i, j in enumerate(res,0):
    line = []
    line.append(str(i))
    line.append(str(' '.join(j)))
    result.append(line)
    
    
f = open('bag_prediction.csv', 'w', encoding = 'big5')
w = csv.writer(f,  quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
w.writerows(result)
f.close()



plt.plot(history_list[0])
plt.plot(history_list[1])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
#plt.show()
plt.savefig('bag_model_loss.png')

plt.clf()


plt.plot(history_list[2])
plt.plot(history_list[4])
plt.plot(history_list[6])
plt.title('model train f1_score')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim([0,1])
plt.legend(['f1_score', 'precision', 'recall'], loc='upper left')
#plt.show()
plt.savefig('bag_train_accuracy.png')

plt.clf()




plt.plot(history_list[3])
plt.plot(history_list[5])
plt.plot(history_list[7])
plt.title('model vailid f1_score')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim([0,1])
plt.legend(['f1_score', 'precision', 'recall'], loc='upper left')
#plt.show()
plt.savefig('bag_valid_accuracy.png')





print(np.max(history_list[3]))







