
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
        stat_list = [1]
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
param_Take = np.insert(param_Take, 0, True)

X = X[:,param_Take]


X = np.asmatrix(X)
y = np.asmatrix(y)



beta = np.dot(X.getT(),X)
beta = beta.getI()
beta = np.dot(beta, X.getT())
beta = np.dot(beta, y.getT())


error = []

k = np.asarray(np.dot(X, beta) - y.getT()).reshape(-1)
error = np.square(k)
    
print("Mean squared error: %.5f"
        % np.mean(error))
        
# remove outliers
error = np.sqrt(error)
iqr = np.percentile(error,75) - np.percentile(error,25)
up_limit = np.percentile(error,75) + 1.5 * iqr
down_limit = np.percentile(error,25) - 1.5 * iqr


while(sum(error >= up_limit) > y.shape[1]*0.01):
    
    print("outliers count: %d"
        %sum(error >= up_limit))  
    
    X = X[(error < up_limit)]
    y = y[:, (error < up_limit)]

    beta = np.dot(X.getT(),X)
    beta = beta.getI()
    beta = np.dot(beta, X.getT())
    beta = np.dot(beta, y.getT())
    
    
    error = []
    
    k = np.asarray(np.dot(X, beta) - y.getT()).reshape(-1)
    error = np.square(k)
        
    print("Mean squared error: %.5f"
            % np.mean(error))

    # remove outliers
    error = np.sqrt(error)
    iqr = np.percentile(error,75) - np.percentile(error,25)
    up_limit = np.percentile(error,75) + 1.5 * iqr
    down_limit = np.percentile(error,25) - 1.5 * iqr
    
    





test_X = []
test_stat = [1]
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
    if len(test_stat) == 9*18 + 1:
        
        
        test_stat = np.asarray(test_stat)
        test_stat[test_stat < 0] = 0
        test_stat = test_stat.tolist()        
        
        
        test_X.append(test_stat)
        test_stat = [1]
f.close()


test_X = np.asarray(test_X)
test_X = test_X[:,param_Take]

res = np.dot(test_X, beta)

result = [['id','value']]
for i, j in zip(test_name, res):
    line = []
    line.append(i)
    if(float(j)>=0):
        line.append(float(j))
    else:
        line.append(float(0))
    result.append(line)

f = open('test_X_result.csv', 'w', encoding = 'big5')
w = csv.writer(f)
w.writerows(result)
f.close()