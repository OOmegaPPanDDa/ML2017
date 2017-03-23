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
        
#        stat_sqrt = [x**0.5 for x in stat_list]
#        stat_list = stat_list + stat_sqrt
        
#        stat_square = [x**2 for x in stat_list]
#        stat_list = stat_list + stat_square
    
#        stat_cube = [x**5 for x in stat_list]
#        stat_list = stat_list + stat_cube
        
        
        X.append(stat_list)
        y.append(label)
        
        #break
    #break
        
        
        
X = np.asarray(X)
y = np.asarray(y)

X = X[y>=0]
y = y[y>=0]



np.random.seed(seed = 5246)
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.8)], indices[int(X.shape[0]*0.8):]




vte_list = []
the_Time_Take = [False, False, True, True, True, True, True, True, True]

"""
for q in range(2**18):
    print("q: %d" % q)
    
    train_X, valid_test_X = X[training_idx,:], X[test_idx,:]
    train_y, valid_test_y = y[training_idx], y[test_idx]
    
    
    train_X = train_X[train_y>=0]
    train_y = train_y[train_y>=0]    
    
    take_bin = '0'*(18 - len(bin(q)[2:])) + bin(q)[2:]
    param_Take = []
    for b in take_bin:
        param_Take.append(bool(int(b)))
        
        
    param_Take = np.repeat(param_Take, 9)
    
    time_Take = the_Time_Take
    timeSum = sum(time_Take)
    time_Take = np.tile(time_Take, 18)
    
    
    #param_Take = np.tile(param_Take, 2)
    #time_Take = np.tile(time_Take, 2)
    
    param_Take = param_Take * time_Take
    param_Take = np.insert(param_Take, 0, True)
    
    train_X = train_X[:, param_Take]
    valid_test_X = valid_test_X[:, param_Take]
    
    
    #iqr = np.percentile(y,75) - np.percentile(y,25)
    #up_limit = np.percentile(y,75) + 1.5 * iqr
    #down_limit = np.percentile(y,25) - 1.5 * iqr
    #
    #X = X[(y>down_limit)*(y < up_limit)]
    #y = y[(y>down_limit)*(y < up_limit)]
    
    train_X = np.asmatrix(train_X)
    train_y = np.asmatrix(train_y)
    valid_test_X = np.asmatrix(valid_test_X)
    valid_test_y = np.asmatrix(valid_test_y)
    
    
    
    beta = np.dot(train_X.getT(),train_X)
    beta = beta.getI()
    beta = np.dot(beta, train_X.getT())
    beta = np.dot(beta, train_y.getT())
    
    
    error = []
    
    k = np.asarray(np.dot(train_X, beta) - train_y.getT()).reshape(-1)
    error = np.square(k)
        
#    print("train Mean squared error: %.5f"
#            % np.mean(error))
            
            
            
    valid_test_error = []
    
    k = np.asarray(np.dot(valid_test_X, beta) - valid_test_y.getT()).reshape(-1)
    valid_test_error = np.square(k)
        
#    print("valid Mean squared error: %.5f"
#            % np.mean(valid_test_error))
            
            
    vte = np.mean(valid_test_error)
    new_vte = vte
            
    # remove outliers
    error = np.sqrt(error)
    iqr = np.percentile(error,75) - np.percentile(error,25)
    up_limit = np.percentile(error,75) + 1.5 * iqr
    down_limit = np.percentile(error,25) - 1.5 * iqr
    
    
    while(new_vte <= vte and sum(error >= up_limit) > 0):
        
#        print("outliers count: %d"
#            %sum(error >= up_limit))
            
        vte = new_vte
        
        train_X = train_X[(error < up_limit)]
        train_y = train_y[:, (error < up_limit)]
    
        beta = np.dot(train_X.getT(),train_X)
        beta = beta.getI()
        beta = np.dot(beta, train_X.getT())
        beta = np.dot(beta, train_y.getT())
        
        
        error = []
    
        k = np.asarray(np.dot(train_X, beta) - train_y.getT()).reshape(-1)
        error = np.square(k)
            
#        print("train Mean squared error: %.5f"
#                % np.mean(error))
                
                
                
        valid_test_error = []
        
        k = np.asarray(np.dot(valid_test_X, beta) - valid_test_y.getT()).reshape(-1)
        valid_test_error = np.square(k)
            
#        print("valid Mean squared error: %.5f"
#                % np.mean(valid_test_error))
        
        new_vte = np.mean(valid_test_error)
    
        # remove outliers
        error = np.sqrt(error)
        iqr = np.percentile(error,75) - np.percentile(error,25)
        up_limit = np.percentile(error,75) + 1.5 * iqr
        down_limit = np.percentile(error,25) - 1.5 * iqr
    
    
    vte_list.append(vte)
    
np.save('vte_list.npy', vte_list)
"""




vte_list = np.load('vte_list.npy')

#q = np.argmin(vte_list)
rank = 0
q = int(np.argwhere(vte_list == np.partition(vte_list, rank)[rank]))
print('q = ', q)


take_bin = '0'*(18 - len(bin(q)[2:])) + bin(q)[2:]
param_Take = []
for b in take_bin:
    param_Take.append(bool(int(b)))
    

#param_Take = [
#        False,  False, False,  False,  False, False, False, False, False,
#        True,  False, False,  False,  False, False, False, False, False]
        

print(param_Take)
print(sum(param_Take))

        
        
param_Take = np.repeat(param_Take, 9)

time_Take = the_Time_Take
timeSum = sum(time_Take)
time_Take = np.tile(time_Take, 18)


#param_Take = np.tile(param_Take, 2)
#time_Take = np.tile(time_Take, 2)

param_Take = param_Take * time_Take
param_Take = np.insert(param_Take, 0, True)

X = X[:, param_Take]


#iqr = np.percentile(y,75) - np.percentile(y,25)
#up_limit = np.percentile(y,75) + 1.5 * iqr
#down_limit = np.percentile(y,25) - 1.5 * iqr
#
#X = X[(y&gt;down_limit)*(y &lt; up_limit)]
#y = y[(y&gt;down_limit)*(y &lt; up_limit)]

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
        
#        test_stat_sqrt = [x**0.5 for x in test_stat]
#        test_stat = test_stat + test_stat_sqrt
        
#        test_stat_square = [x**2 for x in test_stat]
#        test_stat = test_stat + test_stat_square
        
#        test_stat_cube = [x**3 for x in test_stat]
#        test_stat = test_stat + test_stat_cube
        
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


predict = []
f = open('test_X_result.csv','r', encoding = 'big5')
next(f)
for line in f:
    predict.append(float(line[line.find(',')+1:-1]))
f.close()


correct = []

f = open('correct.csv','r', encoding = 'big5')
next(f)
for line in f:
    correct.append(float(line[line.find(',')+1:-1]))
f.close()

del(predict[121])
del(correct[121])
del(predict[192])
del(correct[192])

print(correct)

predict = np.asarray(predict)
correct = np.asarray(correct)

print(((sum((predict - correct)**2)/(240-2))**0.5))






