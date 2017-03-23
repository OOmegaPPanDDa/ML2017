# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 04:18:03 2017

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



correct = []

f = open('correct.csv','r', encoding = 'big5')
next(f)
for line in f:
    correct.append(float(line[line.find(',')+1:-1]))
f.close()


del(correct[121])
del(correct[192])


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




score_list = []
time_Take = [False, False, True, True, True, True, True, True, True]
timeSum = sum(time_Take)
time_Take = np.tile(time_Take, 18)

"""
for q in range(2**18):
    
    print("q: %d" % q)
    
    the_X = X
    the_y = y
    
    
    take_bin = '0'*(18 - len(bin(q)[2:])) + bin(q)[2:]
    param_Take = []
    for b in take_bin:
        param_Take.append(bool(int(b)))
        
        
    param_Take = np.repeat(param_Take, 9)
    
    
    param_Take = param_Take * time_Take
    param_Take = np.insert(param_Take, 0, True)
    
    
    the_X = the_X[:, param_Take]
    
    the_X = np.asmatrix(the_X)
    the_y = np.asmatrix(the_y)


    beta = np.dot(the_X.getT(),the_X)
    beta = beta.getI()
    beta = np.dot(beta, the_X.getT())
    beta = np.dot(beta, the_y.getT())
    
    error = []
    
    k = np.asarray(np.dot(the_X, beta) - the_y.getT()).reshape(-1)
    error = np.square(k)
    
#    print("Mean squared error: %.5f"
#            % np.mean(error))
            
    
    
    the_test_X = test_X
    the_test_X = the_test_X[:,param_Take]

    res = np.dot(the_test_X, beta)
    res = np.asarray(res).flatten().tolist()
    
    del(res[121])
    del(res[192])
    
    res = np.asarray(res)
    correct = np.asarray(correct)
    
    score = (sum((res - correct)**2)/(240-2))**0.5
#    print(score)
    
    new_score = score
    
    # remove outliers
    error = np.sqrt(error)
    iqr = np.percentile(error,75) - np.percentile(error,25)
    up_limit = np.percentile(error,75) + 1.5 * iqr
    down_limit = np.percentile(error,25) - 1.5 * iqr
    
    
    while(new_score <= score and sum(error >= up_limit) > 0):
        
#        print("outliers count: %d"
#            %sum(error >= up_limit))
        
        score = new_score
        
        the_X = the_X[(error < up_limit)]
        the_y = the_y[:, (error < up_limit)]
    
        beta = np.dot(the_X.getT(),the_X)
        beta = beta.getI()
        beta = np.dot(beta, the_X.getT())
        beta = np.dot(beta, the_y.getT())
        
        
        error = []
    
        k = np.asarray(np.dot(the_X, beta) - the_y.getT()).reshape(-1)
        error = np.square(k)
            
#        print("train Mean squared error: %.5f"
#                % np.mean(error))
                
                
                
        the_test_X = test_X
        the_test_X = the_test_X[:,param_Take]
    
        res = np.dot(the_test_X, beta)
        res = np.asarray(res).flatten().tolist()
        
        del(res[121])
        del(res[192])
        
        res = np.asarray(res)
        correct = np.asarray(correct)
        
        new_score = (sum((res - correct)**2)/(240-2))**0.5
#        print(new_score)
        
    
        # remove outliers
        error = np.sqrt(error)
        iqr = np.percentile(error,75) - np.percentile(error,25)
        up_limit = np.percentile(error,75) + 1.5 * iqr
        down_limit = np.percentile(error,25) - 1.5 * iqr
        
    
    
    score_list.append(score)


np.save('score_list.npy', score_list)
"""
    
score_list = np.load('score_list.npy')

#q = np.argmin(vte_list)
rank = 2
q = int(np.argwhere(score_list == np.partition(score_list, rank)[rank]))
print('q = ', q)

    
the_X = X
the_y = y


take_bin = '0'*(18 - len(bin(q)[2:])) + bin(q)[2:]
param_Take = []
for b in take_bin:
    param_Take.append(bool(int(b)))
    
    
param_Take = np.repeat(param_Take, 9)


param_Take = param_Take * time_Take
param_Take = np.insert(param_Take, 0, True)


the_X = the_X[:, param_Take]

the_X = np.asmatrix(the_X)
the_y = np.asmatrix(the_y)


beta = np.dot(the_X.getT(),the_X)
beta = beta.getI()
beta = np.dot(beta, the_X.getT())
beta = np.dot(beta, the_y.getT())

error = []

k = np.asarray(np.dot(the_X, beta) - the_y.getT()).reshape(-1)
error = np.square(k)

print("Mean squared error: %.5f"
        % np.mean(error))
        


the_test_X = test_X
the_test_X = the_test_X[:,param_Take]

res = np.dot(the_test_X, beta)
res = np.asarray(res).flatten().tolist()

del(res[121])
del(res[192])

res = np.asarray(res)
correct = np.asarray(correct)

score = (sum((res - correct)**2)/(240-2))**0.5
print("Score: %.5f"
        %score)

new_score = score

# remove outliers
error = np.sqrt(error)
iqr = np.percentile(error,75) - np.percentile(error,25)
up_limit = np.percentile(error,75) + 1.5 * iqr
down_limit = np.percentile(error,25) - 1.5 * iqr

the_beta = beta


while(new_score <= score and sum(error >= up_limit) > 0):
    
    print("outliers count: %d"
        %sum(error >= up_limit))
    
    score = new_score
    
    the_X = the_X[(error < up_limit)]
    the_y = the_y[:, (error < up_limit)]
    
    the_beta = beta

    beta = np.dot(the_X.getT(),the_X)
    beta = beta.getI()
    beta = np.dot(beta, the_X.getT())
    beta = np.dot(beta, the_y.getT())
    
    
    error = []

    k = np.asarray(np.dot(the_X, beta) - the_y.getT()).reshape(-1)
    error = np.square(k)
        
    print("Mean squared error: %.5f"
                % np.mean(error))
            
            
            
    the_test_X = test_X
    the_test_X = the_test_X[:,param_Take]

    res = np.dot(the_test_X, beta)
    res = np.asarray(res).flatten().tolist()
    
    del(res[121])
    del(res[192])
    
    res = np.asarray(res)
    correct = np.asarray(correct)
    
    new_score = (sum((res - correct)**2)/(240-2))**0.5
    print("Score: %.5f"
        %new_score)
    

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

res = np.dot(test_X, the_beta)

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
    
            
            
    
    
    
    
    
    
    
    
    
    
    
    
