from keras.models import load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import csv

nb_class = 7


dataset_label = []
dataset_feature = []

f = open('train.csv', 'r')

reader = csv.reader(f)
f.readline()
for line in f:
    the_label = int(line.split(',')[0])
    dataset_label.append(the_label)
    the_feature = line[line.find(',') + 1:].split(' ')
    the_feature = list(map(int, the_feature))    
    dataset_feature.append(the_feature)    
f.close()

dataset_label = np.asarray(dataset_label)
dataset_feature = np.asarray(dataset_feature)

dataset_label = to_categorical(dataset_label, nb_class)

dataset_feature = dataset_feature.reshape(dataset_feature.shape[0], 48, 48, 1)

dataset_feature = dataset_feature.astype('float32')


np.random.seed(seed = 4646)

train_valid_ratio = 0.9
indices = np.random.permutation(dataset_feature.shape[0])
train_idx, valid_idx = indices[:int(dataset_feature.shape[0] * train_valid_ratio)], indices[int(dataset_feature.shape[0] * train_valid_ratio):]
train_feature, valid_feature = dataset_feature[train_idx,:], dataset_feature[valid_idx,:]
train_label, valid_label = dataset_label[train_idx,:], dataset_label[valid_idx,:]


model = load_model('model.h5')
input_img = model.input

test_id = []
test_feature = []

f = open('test.csv', 'r')

reader = csv.reader(f)
f.readline()
for line in f:
    the_id = int(line.split(',')[0])
    test_id.append(the_id)
    the_feature = line[line.find(',') + 1:].split(' ')
    the_feature = list(map(int, the_feature))    
    test_feature.append(the_feature)    
f.close()

test_feature = np.asarray(test_feature)

test_feature = test_feature.reshape(test_feature.shape[0], 48, 48 , 1)

test_img = test_feature[1]
