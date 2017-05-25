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
import seaborn as sns


from sklearn.preprocessing import MultiLabelBinarizer




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




mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)


y_train = y_train.astype('float32')

print(list(mlb.classes_))




plt.figure(figsize=(15,15))
sns.heatmap(np.corrcoef(np.transpose(y_train)), 
        xticklabels=mlb.classes_,
        yticklabels=mlb.classes_)
        

plt.tight_layout()
plt.savefig('tag_heatmap.png', dpi = 800)




plt.clf()
tag_sum = np.sum(y_train, axis=0)
print(tag_sum)

tags = list(zip(tag_sum, mlb.classes_))

tags = sorted(tags, reverse= True)

tag_freq = []
tag_name = []

for tag in tags:
    tag_freq.append(tag[0])
    tag_name.append(tag[1])
    

plt.figure(figsize=(10,8))
plt.bar(range(len(tag_freq)), tag_freq, 0.8, color="deepskyblue", align='center')
plt.xticks(range(len(tag_name)), tag_name, rotation='vertical')
plt.tight_layout()
plt.savefig('tag_bar.png', dpi = 800)

