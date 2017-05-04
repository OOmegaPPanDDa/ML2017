from sklearn.metrics import classification_report,confusion_matrix
from keras.models import load_model
from keras.utils.np_utils import to_categorical
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


y_pred = model.predict_classes(valid_feature)
print(y_pred)

target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
class_report = classification_report(np.argmax(valid_label, axis=1), y_pred, target_names=target_names)
print(class_report)
conf_mat = confusion_matrix(np.argmax(valid_label, axis=1), y_pred)
print(conf_mat)
row_sums = conf_mat.sum(axis=1)
conf_mat = conf_mat/row_sums[:, np.newaxis]
conf_mat = np.around(conf_mat, decimals=2)
print(conf_mat)

norm_conf = []
for i in conf_mat:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

width, height = conf_mat.shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf_mat[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
plt.xticks(range(width), target_names)
plt.yticks(range(height), target_names)
plt.ylabel('True')
plt.xlabel('Predict')
plt.savefig('confusion_matrix.png', format='png')