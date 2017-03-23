# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:09:10 2017

@author: HSIN
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

data = []

f = open('./data/train.csv', 'r', encoding = 'big5')
#f = open(sys.argv[1], 'r', encoding = 'big5')

# skip the first line
train_title = f.readline()

for line in csv.reader(f):
    
    # get data from column index 3 and after
    # replace NR by 0 
    the_data = [float(i.replace('NR','0')) for i in line[3:]]
    
    data.append(the_data)
    
f.close()




X = []
y = []


# 資料是不完全連續的，每個月只有前20天
# 所以用 for loop 來取每個月中的資料

for month in range(1,13):
    
    # 每月資料有 18 feature * 20 day =  360 筆 data
    month_data = data[360 * (month-1):360 * month]
    
    
    # 建立 feature_list[0] ~ feature_list[18]
    # feature_list[i] 有  24 hr * 20 day = 480 筆資料
    
    feature_id = list(range(18))
    
    feature_list = []
    
    for fid in range(18):
        feature_list.append([])
        
    for i, row in enumerate(month_data):
        the_fid = i % 18
        for stat in row:
            feature_list[the_fid].append(stat)
            
            
    # PM2.5 在 index 9
    # 因為是用前 9 小時資料來預測 PM2.5
    # 因此每個月前 9 筆  PM2.5 不能作為 training data
            
    for j, label in enumerate(feature_list[9][9:], 9):
        take_to_train = True
        stat_list = []
        # loop 18 features
        for fid in feature_id:
            # loop prev 9 hr
            for feature in feature_list[fid][j-9:j]:
                
                # PM2.5 資料中 < 0 的資料多為測站有誤
                # 不取錯誤的 traininng data
                if fid == 9 and feature < 0:
                   take_to_train = False
                   break
                else:
                    stat_list.append(feature)
                
        

        
        if(take_to_train):
            
            the_stat_list = stat_list
            
#            stat_sqrt = [x**0.5 for x in the_stat_list]
#            stat_list = stat_list + stat_sqrt
            
            stat_square = [x**2 for x in the_stat_list]
            stat_list = stat_list + stat_square
#            stat_list = stat_square
 
            stat_cube = [x**3 for x in the_stat_list]
            stat_list = stat_list + stat_cube
#            stat_list = stat_cube
            
            
#            stat_list = stat_square + stat_cube    
            
            
            X.append(stat_list)
            y.append(label)
        
        
X = np.asarray(X)
y = np.asarray(y)


# PM2.5 資料中 < 0 的資料多為測站有誤
# 不取錯誤的 traininng data

X = X[y >= 0]
y = y[y >= 0]

record = []
#record.append([, 'train_RMSE','valid_RMSE'])

# train test split
np.random.seed(seed = 524617)

train_valid_ratio = 0.5
indices = np.random.permutation(X.shape[0])
train_idx, valid_idx = indices[:int(X.shape[0] * train_valid_ratio)], indices[int(X.shape[0] * train_valid_ratio):]
the_train_X, valid_X = X[train_idx,:], X[valid_idx,:]
the_train_y, valid_y = y[train_idx], y[valid_idx]



    
train_volume_ratio = 1
indices = np.random.permutation(the_train_X.shape[0])
train_idx = indices[:int(the_train_X.shape[0] * train_volume_ratio)]
train_X = the_train_X[train_idx,:]
train_y = the_train_y[train_idx]

train_volume = len(train_y)

feature_Take = [
        False,  False, False,  False,  False, False, False, False, False,
        True,  False, False,  False,  False, False, False, False, False]


time_Take = [True, True, True, True, True, True, True, True, True]
timeSum = sum(time_Take)


# 總共 18 feature * 9 hr = 162 parameters
param_Take = np.repeat(feature_Take, 9) * np.tile(time_Take, 18)




param_Take = np.tile(param_Take, 3)
timeSum = timeSum * 3


lamb = 0.005
max_iteration = 10000


initial_b = 1
initial_w = -0.01

lr = 1

b = initial_b
b_lr = 0.0

w_vector = np.zeros(X.shape[1], dtype = np.float) + initial_w

#pm2.5 should be very relative
w_vector[81:90] = np.zeros(9, dtype = np.float) + 1/timeSum

w_lr_vector = np.zeros(X.shape[1], dtype = np.float)




x_Take = train_X[:, param_Take]

w_vector = w_vector[param_Take]
w_lr_vector = w_lr_vector[param_Take]


train_mse = float("inf")
valid_mse = float("inf")

# Iteration training
for i in range(max_iteration):
    
    
    b_grad = 0.0
    w_grad_vector = np.zeros(X.shape[1], dtype=np.float)
    w_grad_vector = w_grad_vector[param_Take]
    
    
    for n in range(len(train_y)):
        x = x_Take[n]
        
        e = train_y[n] - b - sum(w_vector * x)
        e += lamb * sum(w_vector ** 2)
        
        b_grad = b_grad  - 2.0 * e * 1.0
        w_grad_vector = w_grad_vector  - 2.0 * e * x
            
    
    # Adagrad
    b_lr += b_grad ** 2
    w_lr_vector += w_grad_vector ** 2
    
    # Update parameters.
    # eps = 1e-8
    b -= lr * b_grad / (np.sqrt(b_lr))
    w_vector -= lr * w_grad_vector / (np.sqrt(w_lr_vector))
    
    
    
    if ((i+1)%10 == 0):
        
        
        
        
        train_error = []
        for y_index in range(len(train_y)):
            x = x_Take[y_index]
            train_error.append((train_y[y_index] - b - sum(w_vector * x))**2)
            
            
            
            
            
        valid_error = []
        for y_index in range(len(valid_y)):
            x = valid_X[:, param_Take][y_index]
            valid_error.append((valid_y[y_index] - b - sum(w_vector * x))**2)
            
            
            
        if(np.mean(valid_error) > valid_mse):
            break
        
        else:
            train_mse = np.mean(train_error)
            valid_mse = np.mean(valid_error)
            
            
        print(
            "Iteration Times: %d" 
            % (i+1),
            "train RMSE: %.5f"
            % np.mean(train_error)** 0.5,
            "valid RMSE: %.5f"
            % np.mean(valid_error)** 0.5
            )
            
   


best_b = b
best_w_vector = w_vector


remove_outliers = False



if(remove_outliers):

    train_error = []          
    for y_index in range(len(train_y)):
        x = x_Take[y_index]
        train_error.append((train_y[y_index] - b - sum(w_vector * x))**2)
        
        
    print("train_MSE: %.5f"
            % np.mean(train_error))
                  
    print("train_RMSE: %.5f"
              % np.mean(train_error)** 0.5)
            
    # outliers
    train_error = np.sqrt(train_error)
    iqr = np.percentile(train_error, 75) - np.percentile(train_error, 25)
    up_limit = np.percentile(train_error, 75) + 1.5 * iqr
    down_limit = np.percentile(train_error, 25) - 1.5 * iqr
    
    
    
    
    
    
    # 如果 outliers 數量大於總體的 1%
    # 則移除 outliers 並重新 train
    while(sum(train_error >= up_limit) > len(train_y) * 0.01):
        
    
        
        print("outliers count: %d"
            %sum(train_error >= up_limit))  
            
            
        best_b = b
        best_w_vector = w_vector
        
        # remove upper outliers
        train_X = train_X[(train_error < up_limit)]
        train_y = train_y[(train_error < up_limit)]
        
        
        
        # re-train
        
        initial_b = 1
        initial_w = -0.01
        
        lr = 1
        
        
        b = initial_b
        b_lr = 0.0
        
        w_vector = np.zeros(X.shape[1], dtype = np.float) + initial_w
        
        #pm2.5 should be very relative
        w_vector[81:90] = np.zeros(9, dtype = np.float) + 1/timeSum
        
        w_lr_vector = np.zeros(X.shape[1], dtype = np.float)
        
        
        
        
        x_Take = train_X[:, param_Take]
        
        w_vector = w_vector[param_Take]
        w_lr_vector = w_lr_vector[param_Take]
        
        
        train_mse = float("inf")
        valid_mse = float("inf")
        
        
        # Iterations
        for i in range(max_iteration):
            
            
            b_grad = 0.0
            w_grad_vector = np.zeros(X.shape[1], dtype=np.float)
            w_grad_vector = w_grad_vector[param_Take]
            
            
            
            for n in range(len(train_y)):
                x = x_Take[n]
                
                e = train_y[n] - b - sum(w_vector * x)
                e += lamb * sum(w_vector ** 2)
                
                b_grad = b_grad  - 2.0 * e * 1.0
                w_grad_vector = w_grad_vector  - 2.0 * e * x
                    
            
            # Adagrad
            b_lr += b_grad ** 2
            w_lr_vector += w_grad_vector ** 2
            
            # Update parameters.
            # eps = 1e-8
            b -= lr * b_grad / (np.sqrt(b_lr))
            w_vector -= lr * w_grad_vector / (np.sqrt(w_lr_vector))
            
                      
            
            
            
        
        
        
            if ((i+1)%10 == 0):
                
                train_error = []
                for y_index in range(len(train_y)):
                    x = x_Take[y_index]
                    train_error.append((train_y[y_index] - b - sum(w_vector * x))**2)
                    
                
                valid_error = []
                for y_index in range(len(valid_y)):
                    x = valid_X[:, param_Take][y_index]
                    valid_error.append((valid_y[y_index] - b - sum(w_vector * x))**2)
                    
                 
                 
                 
                if(np.mean(valid_error) > valid_mse):                
                    break
        
                else:
                    train_mse = np.mean(train_error)
                    valid_mse = np.mean(valid_error)
                    
                
                print(
                "Iteration Times: %d" 
                % (i+1),
                "train RMSE: %.5f"
                % np.mean(train_error)** 0.5,
                "valid RMSE: %.5f"
                % np.mean(valid_error)** 0.5
                )
                 
            
                  
                  
                  
        train_error = []          
        for y_index in range(len(train_y)):
            x = x_Take[y_index]
            train_error.append((train_y[y_index] - b - sum(w_vector * x))**2)
    
            
        # outliers
        train_error = np.sqrt(train_error)
        iqr = np.percentile(train_error,75) - np.percentile(train_error,25)
        up_limit = np.percentile(train_error,75) + 1.5 * iqr
        down_limit = np.percentile(train_error,25) - 1.5 * iqr




param_Take_len = len(param_Take)


# save model
param = np.append(param_Take, [best_b])
param = np.append(param, best_w_vector)
np.save('the_model.npy', param)


# load model
param = np.load('the_model.npy')
param_Take = np.asarray(param[0:param_Take_len], dtype=bool)
b = param[param_Take_len]
w_vector = param[param_Take_len + 1:]






test_X = []
test_stat = []
test_name = []

f = open('./data/test_X.csv', 'r', encoding = 'big5')
#f = open(sys.argv[2], 'r', encoding = 'big5')

for line in csv.reader(f):
    
    # test number index
    if line[0] not in test_name:
        test_name.append(line[0])
        
    for stat in line[2:]:
        if stat != 'NR':
            test_stat.append(float(stat))
        # replace NR by 0
        else:
            test_stat.append(float(0))
            
            
    
    if len(test_stat) == 9*18:
        
        the_test_stat = test_stat
        
#        test_stat_sqrt = [x**0.5 for x in the_test_stat]
#        test_stat = test_stat + test_stat_sqrt
        
        test_stat_square = [x**2 for x in the_test_stat]
        test_stat = test_stat + test_stat_square
#        test_stat = test_stat_square
        
        test_stat_cube = [x**3 for x in the_test_stat]
        test_stat = test_stat + test_stat_cube
#        test_stat = test_stat_cube
        
        
#        test_stat = test_stat_square + test_stat_cube
        
        test_X.append(test_stat)
        test_stat = []
f.close()




test_X = np.asarray(test_X)
test_X_Take = test_X[:,param_Take]
label_test_X = []


iqr = np.percentile(y, 75) - np.percentile(y, 25)
y_up_limit = np.percentile(y, 75) + 1.5 * iqr
y_down_limit = np.percentile(y, 25) - 1.5 * iqr

for stat in test_X_Take:
    x = stat
    if b + sum(w_vector * x) >= y_down_limit:
        if b + sum(w_vector * x) <= y_up_limit:
            label_test_X.append(b + sum(w_vector * x))
        else:
            label_test_X.append(y_up_limit)
    else:
        label_test_X.append(y_down_limit)
    
result = [['id','value']]
for i, j in zip(test_name, label_test_X):
    line = []
    line.append(i)
    line.append(j)
    result.append(line)
    
    
    

f = open('./result/res.csv', 'w', encoding = 'big5')
#f = open(sys.argv[3], 'w', encoding = 'big5')
w = csv.writer(f)
w.writerows(result)
f.close()
