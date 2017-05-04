import sys
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta
from keras.models import load_model
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle


nb_class = 7
batch_size = 128
epochs = 1500
patience = 100


model = load_model('model.h5')


test_id = []
test_feature = []

#f = open('test.csv', 'r')
f = open(sys.argv[1], 'r', encoding = 'big5')

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

res = model.predict_classes(test_feature, batch_size=batch_size)

result = [['id','label']]
for i, j in enumerate(res):
  line = []
  line.append(i)
  line.append(j)
  result.append(line)

#f = open('prediction.csv', 'w')
f = open(sys.argv[2], 'w')
w = csv.writer(f)
w.writerows(result)
f.close()