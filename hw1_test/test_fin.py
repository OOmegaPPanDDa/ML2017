# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:00:08 2017

@author: HSIN
"""

import csv
import numpy as np

data = []

f = open('train.csv', 'r', encoding = 'big5')
first_line = f.readline()
for line in csv.reader(f):
    the_data = [float(i.replace('NR','0')) for i in line[3:]]
    data.append(the_data)
f.close()

X = []
y = []


for month in range(1,13):
    month_data = data[360 * (month-1):360 * month]
    #print(len(month_data))
    
    feature_id = list(range(18))
    #print(feature_id)
    
    feature_list = []
    for fid in range(18):
        feature_list.append([])
        
    for i, row in enumerate(month_data):
        the_fid = i%18
        for stat in row:
            feature_list[the_fid].append(stat)
            
    for j, label in enumerate(feature_list[9][9:],9):
        stat_list = []
        for fid in feature_id:
            for feature in feature_list[fid][j-9:j]:
                stat_list.append(feature)

        stat_list = np.asarray(stat_list)
        stat_list[stat_list < 0] = 0
        stat_list = stat_list.tolist()

        
        
        X.append(stat_list)
        y.append(label)
        
        #break
    #break
        
        
        
X = np.asarray(X)
y = np.asarray(y)


X = X[y>=0]
y = y[y>=0]



param_Take = [
        False,  False, False,  False,  False, False, False, False, False,
        True,  False, False,  False,  False, False, False, False, False]
        
        
param_Take = np.repeat(param_Take, 9)

time_Take = [False, False, True, True, True, True, True, True, True]
timeSum = sum(time_Take)
time_Take = np.tile(time_Take, 18)


#param_Take = np.tile(param_Take, 2)
#time_Take = np.tile(time_Take, 2)

param_Take = param_Take * time_Take


initial_b = 1
initial_w = -0.1

lr = 1
iteration = 3000


b = initial_b
b_lr = 0.0

w_vector = np.zeros(X.shape[1], dtype=np.float) + initial_w
w_vector[81:90] = np.zeros(9, dtype=np.float) + 1/timeSum
#pm2.5 should be very relative
w_lr_vector = np.zeros(X.shape[1], dtype=np.float)



w_vector = w_vector[param_Take]
w_lr_vector = w_lr_vector[param_Take]

x_Take = X[:,param_Take]

# Iterations
for i in range(iteration):
    
    
    b_grad = 0.0
    w_grad_vector = np.zeros(X.shape[1], dtype=np.float)
    
    
    w_grad_vector = w_grad_vector[param_Take]
    
    
    
    for n in range(len(y)):
        x = x_Take[n]
        
        e = y[n] - b - sum(w_vector * x)
        
        b_grad = b_grad  - 2.0 * e * 1.0
        for m in range(len(w_grad_vector)):
            w_grad_vector[m] = w_grad_vector[m]  - 2.0 * e * x[m]
            
            
    b_lr += b_grad**2
    w_lr_vector += w_grad_vector**2
    
    # Update parameters.
    eps = 1e-8
    b += - lr * b_grad / (np.sqrt(b_lr) + eps)
    w_vector += - lr * w_grad_vector / (np.sqrt(w_lr_vector) + eps)
    
    
    
    
    if (i+1)%10 == 0:
        
        print(i+1)
        error = []
        for i in range(len(y)):
            x = x_Take[i]
            error.append((y[i] - b - sum(w_vector * x))**2)
            
        print("Mean squared error: %.5f"
              % np.mean(error))



param_Take_len = len(param_Take)

param = np.append(param_Take,[b])
param = np.append(param, w_vector)
np.save('model.npy', param)

param = np.load('model.npy')
param_Take = np.asarray(param[0:param_Take_len], dtype=bool)
b = param[param_Take_len]
w_vector = param[param_Take_len+1:]

error = []          
for i in range(len(y)):
    x = x_Take[i]
    error.append((y[i] - b - sum(w_vector * x))**2)
    
    
print("Mean squared error: %.5f"
        % np.mean(error))
        
# remove outliers
error = np.sqrt(error)
iqr = np.percentile(error,75) - np.percentile(error,25)
up_limit = np.percentile(error,75) + 1.5 * iqr
down_limit = np.percentile(error,25) - 1.5 * iqr





while(sum(error >= up_limit) > len(y)*0.01):
    
    print("outliers count: %d"
        %sum(error >= up_limit))  
    
    X = X[(error < up_limit)]
    y = y[(error < up_limit)]
    
    
    
    # retrain
    
    x_Take = X[:,param_Take]
    
    # Iterations
    for i in range(iteration):
        
        
        b_grad = 0.0
        w_grad_vector = np.zeros(X.shape[1], dtype=np.float)
        
        
        w_grad_vector = w_grad_vector[param_Take]
        
        
        
        for n in range(len(y)):
            x = x_Take[n]
            
            e = y[n] - b - sum(w_vector * x)
            
            b_grad = b_grad  - 2.0 * e * 1.0
            for m in range(len(w_grad_vector)):
                w_grad_vector[m] = w_grad_vector[m]  - 2.0 * e * x[m]
                
                
        b_lr += b_grad**2
        w_lr_vector += w_grad_vector**2
        
        # Update parameters.
        eps = 1e-8
        b += - lr * b_grad / (np.sqrt(b_lr) + eps)
        w_vector += - lr * w_grad_vector / (np.sqrt(w_lr_vector) + eps)
        
        
        
        
        if (i+1)%10 == 0:
            
            print(i+1)
            error = []
            for i in range(len(y)):
                x = x_Take[i]
                error.append((y[i] - b - sum(w_vector * x))**2)
                
            print("Mean squared error: %.5f"
                  % np.mean(error))
    
    
    
    param_Take_len = len(param_Take)
    
    param = np.append(param_Take,[b])
    param = np.append(param, w_vector)
    np.save('model.npy', param)
    
    param = np.load('model.npy')
    param_Take = np.asarray(param[0:param_Take_len], dtype=bool)
    b = param[param_Take_len]
    w_vector = param[param_Take_len+1:]
    
    error = []          
    for i in range(len(y)):
        x = x_Take[i]
        error.append((y[i] - b - sum(w_vector * x))**2)
        
    print("Mean squared error: %.5f"
            % np.mean(error))
            
    # remove outliers
    error = np.sqrt(error)
    iqr = np.percentile(error,75) - np.percentile(error,25)
    up_limit = np.percentile(error,75) + 1.5 * iqr
    down_limit = np.percentile(error,25) - 1.5 * iqr


test_X = []
test_stat = []
test_name = []
f = open('test_X.csv', 'r', encoding = 'big5')
for line in csv.reader(f):
    if line[0] not in test_name:
        test_name.append(line[0])
    for stat in line[2:]:
        if stat != 'NR':
            test_stat.append(float(stat))
        else:
            test_stat.append(float(0))
    if len(test_stat) == 9*18:
        
        
        test_stat = np.asarray(test_stat)
        test_stat[test_stat < 0] = 0
        test_stat = test_stat.tolist()        
        

        test_X.append(test_stat)
        test_stat = []
f.close()

test_X = np.asarray(test_X)
test_X_Take = test_X[:,param_Take]
label_test_X = []

for stat in test_X_Take:
    x = stat
    if b + sum(w_vector * x) > 0:
        label_test_X.append(b + sum(w_vector * x))
    else:
        label_test_X.append(0)
    
result = [['id','value']]
for i, j in zip(test_name, label_test_X):
    line = []
    line.append(i)
    line.append(j)
    result.append(line)

f = open('test_X_result.csv', 'w', encoding = 'big5')
w = csv.writer(f)
w.writerows(result)
f.close()

