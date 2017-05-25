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





test_texts = []


# f = open('test_data.csv','r')
f = open(sys.argv[1])
test_header = f.readline()
for row in f:
    index = row[:row.find(',')]
    rest_row = row[row.find(',')+1:]
    

    # text = rest_row
    
    text = rest_row.translate(translator)
    text = [i for i in text.lower().split() if i not in stop]
    # text_len.append(len(text))
    text = ' '.join(text)
 
    test_texts.append(text)
    
f.close()



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




maxlen = 186
print(maxlen)


test_sequences = tokenizer.texts_to_sequences(test_texts)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

with open('mlb','rb') as fp:
    mlb = pickle.load(fp)

print(list(mlb.classes_))


embedding_dict = {}
word_vec_dim = 100




batch_size = 128
thresh = 0.4

model = load_model('rnn_complete_model.h5')


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
    
    
# f = open('rnn_prediction.csv', 'w', encoding = 'big5')
f = open(sys.argv[2], 'w', encoding = 'big5')
w = csv.writer(f,  quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
w.writerows(result)
f.close()


"""
plt.plot(history_list[0])
plt.plot(history_list[1])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
#plt.show()
plt.savefig('rnn_model_loss.png')

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
plt.savefig('rnn_train_accuracy.png')

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
plt.savefig('rnn_valid_accuracy.png')





print(np.max(history_list[3]))
"""