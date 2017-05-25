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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


from keras.models import load_model





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
batch_size = 128
bag_voter_num = 10
rnn_voter_num = 10


res_box = []

test_texts = []
# f = open('test_data.csv','r')
f = open(sys.argv[1],'r')
test_header = f.readline()
for row in f:
    index = row[:row.find(',')]
    rest_row = row[row.find(',')+1:]

    text = rest_row.translate(translator)
    text = [i for i in text.lower().split() if i not in stop]
    text = ' '.join(text)
 
    test_texts.append(text)
    
f.close()



tokenizer = pickle.load(open('tokenizer', 'rb'))


word_index = tokenizer.word_index

print(len(tokenizer.word_index))


#################################
# Bag
#################################

bag_mode = 'tfidf'

X_test = tokenizer.texts_to_matrix(test_texts, mode=bag_mode)

with open('mlb','rb') as fp:
    mlb = pickle.load(fp)

X_test = X_test.astype('float32')

print(list(mlb.classes_))





        

for voter in range(bag_voter_num):
    
    print('bag voter number: ', voter)


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





#################################
# RNN
#################################


maxlen = 186
print(maxlen)

test_sequences = tokenizer.texts_to_sequences(test_texts)
X_test = pad_sequences(test_sequences, maxlen=maxlen)

X_test = X_test.astype('float32')


print(list(mlb.classes_))



for voter in range(rnn_voter_num):
    
    print('rnn voter number: ', voter)

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


    



res_box = np.asarray(res_box)
print('res_box_shape', res_box.shape)

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
    
    
# f = open('prediction_check.csv', 'w', encoding = 'big5')
f = open(sys.argv[2], 'w', encoding = 'big5')
w = csv.writer(f,  quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
w.writerows(result)
f.close()

