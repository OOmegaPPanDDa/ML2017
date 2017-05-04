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

dataset_label = []
dataset_feature = []

#f = open('train.csv', 'r')
f = open(sys.argv[1], 'r', encoding = 'big5')

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

"""
train_label = dataset_label 
train_feature = dataset_feature
"""

"""
valid_label = dataset_label
valid_feature = dataset_feature
"""


model = Sequential()
model.add(Conv2D(64, (5 , 5), input_shape=(48, 48, 1)))
model.add(PReLU())
model.add(ZeroPadding2D(padding=(2, 2)))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
  
model.add(ZeroPadding2D(padding=(1, 1))) 
model.add(Conv2D(64, (3, 3)))
model.add(PReLU())
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(64, (3, 3)))
model.add(PReLU())
model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
 
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(PReLU())
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(PReLU())
 
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
 
model.add(Flatten())
model.add(Dense(1024))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(PReLU())
model.add(Dropout(0.2))

 
model.add(Dense(nb_class))
model.add(Activation('softmax'))

ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
        optimizer=ada,
        metrics=['accuracy'])
model.summary()



callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
    History()
]


datagen = ImageDataGenerator(
    width_shift_range=0.5,
    height_shift_range=0.5,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)


datagen.fit(train_feature)



history = model.fit_generator(
  datagen.flow(x = train_feature,y = train_label,
            batch_size=batch_size),
  steps_per_epoch=int(len(train_feature)/batch_size), 
  epochs=epochs,
  validation_data=(valid_feature, valid_label),
  callbacks=callbacks)


model.save('model.h5')