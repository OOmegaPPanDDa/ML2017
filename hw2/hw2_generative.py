# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:06:00 2017

@author: HSIN
"""


import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

y = []
#f = open('Y_train', 'r')
f = open(sys.argv[1], 'r')
for line in f:
    y.append(int(line))
f.close()


dataset = []

#f = open('X_train', 'r')
f = open(sys.argv[2], 'r')
title = f.readline()

reader = csv.reader(f)

for row in reader:
    row_num = [1]
    row_line = list(row)
    
    for param_1d in row_line:
        row_num.append(int(param_1d))
        
    for param_2d in row_line:
        row_num.append(int(param_2d)**2)
        
    for param_3d in row_line:
        row_num.append(int(param_3d)**3)
        
    dataset.append(row_num)


dataset = np.asarray(dataset)



mean = np.mean(dataset, axis=0)
mean[0] = 0
train_mean = np.tile(mean,(len(dataset),1))

std = np.std(dataset, axis=0)
std[0] = 1
train_std = np.tile(std,(len(dataset),1))

dataset = dataset - train_mean
dataset = dataset/train_std


dataset = np.array(list(zip(dataset, y)))
f.close()

param_take_num = len(row_num)


np.random.seed(seed = 46)

train_valid_ratio = 0.8
indices = np.random.permutation(dataset.shape[0])
train_idx, valid_idx = indices[:int(dataset.shape[0] * train_valid_ratio)], indices[int(dataset.shape[0] * train_valid_ratio):]
train_dataset, valid_dataset = dataset[train_idx,:], dataset[valid_idx,:]



def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    


def gradient(dataset, w):
    g = np.zeros(len(w))
    for x,y in dataset:
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        g[0] -= 2 * (y - error) * 1
        g[1:] -= 2 * (y - error) * x[1:]
    return g


#計算現在的權重的錯誤有多少

def cost(dataset, w):
    total_cost = 0
    for x,y in dataset:
        x = np.array(x)
        yhat = sigmoid(w.T.dot(x))
        total_cost += abs(y - yhat)
    return total_cost


def acc(dataset, w):
    error = 0
    for x,y in dataset:
        x = np.array(x)
        yhat = round(sigmoid(w.T.dot(x)))
        error += abs(y - yhat)
    return 1 - error/len(dataset)
    
    
def predict(dataset, w):
    y_test = []
    for x in dataset:
        y_test.append(int(round(sigmoid(w.T.dot(x)))))
    return y_test

def logistic(train_dataset, valid_dataset):

    w = np.zeros(param_take_num) 

    limit = 3000
    plot_x = limit

    eta = 1 #更新幅度
    lr = np.zeros(len(w))

    costs = []
    train_accs = []
    valid_accs = []
    w_record = []

    for i in range(limit):
        current_cost = cost(train_dataset, w)
        train_accuracy = acc(train_dataset, w)
        valid_accuracy = acc(valid_dataset, w)
        
#        print (i, 
#               "cost=", current_cost, 
#               "tacc=", train_accuracy, 
#               "vacc=", valid_accuracy)
               
        costs.append(current_cost)
        train_accs.append(train_accuracy)
        valid_accs.append(valid_accuracy)
        w_record.append(w)
        
        valid_dist = 2000
        if(i >= valid_dist and valid_accs[i] < valid_accs[i - valid_dist]):
            plot_x = i + 1
            break
        
        
#        w = w - eta * gradient(train_dataset, w)
        
        # Adagrad
        
        lr += gradient(train_dataset, w) ** 2
        eps = 1e-8
        w -= eta * gradient(train_dataset, w) / (np.sqrt(lr) + eps)
        
#        eta *= 0.9995 #更新幅度，逐步遞減


#    plt.plot(range(plot_x), costs)
#    plt.show()
#    
#    plt.plot(range(plot_x), train_accs, range(plot_x), valid_accs)
#    plt.show()
    
#    print('best_index = ', np.argmax(valid_accs))
#    print('best_vacc =', valid_accs[np.argmax(valid_accs)])
    w = w_record[np.argmax(valid_accs)]
    
    
    return w
    
    
    
"""
w = logistic(train_dataset, valid_dataset)

np.save('generative_model', w)
"""


w = np.load('generative_model.npy')

test_dataset = []

#f = open('X_test', 'r')
f = open(sys.argv[3], 'r')
title = f.readline()

reader = csv.reader(f)

for row in reader:
    row_num = [1]
    row_line = list(row)
    
    for param_1d in row_line:
        row_num.append(int(param_1d))
        
    for param_2d in row_line:
        row_num.append(int(param_2d)**2)
        
    for param_3d in row_line:
        row_num.append(int(param_3d)**3)
        
    test_dataset.append(row_num)


test_dataset = np.asarray(test_dataset)

test_mean = np.tile(mean,(len(test_dataset),1))
test_std = np.tile(std,(len(test_dataset),1))

test_dataset = test_dataset - test_mean
test_dataset = test_dataset/test_std

f.close()

res = predict(test_dataset, w)

result = [['id','label']]
for i, j in enumerate(res,1):
    line = []
    line.append(i)
    line.append(j)
    result.append(line)
    
    
#f = open('prediction.csv', 'w', encoding = 'big5')
f = open(sys.argv[4], 'w', encoding = 'big5')
w = csv.writer(f)
w.writerows(result)
f.close()