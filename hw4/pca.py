# -*- coding: utf-8 -*-
"""
Created on Sun May 14 06:12:33 2017

@author: HSIN
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dataset = []

"""
for i in range(ord('A'), ord('M')+1):
    for j in range(75):
        if(j>=10):
            print(chr(i)+str(j)+str(".bmp"))
            im = Image.open(chr(i)+str(j)+str(".bmp"))
        else:
            print(chr(i)+str('0')+str(j)+str(".bmp"))
            im = Image.open(chr(i)+str('0')+str(j)+str(".bmp"))
        
        dataset.append(np.array(im.getdata()))
"""




for i in range(ord('A'), ord('A')+10):
    for j in range(10):
        if(j>=10):
            print(chr(i)+str(j)+str(".bmp"))
            im = Image.open('./faceExpressionDatabase/'+chr(i)+str(j)+str(".bmp"))
        else:
            print(chr(i)+str('0')+str(j)+str(".bmp"))
            im = Image.open('./faceExpressionDatabase/'+chr(i)+str('0')+str(j)+str(".bmp"))
        
        dataset.append(np.array(im.getdata()))


dataset = np.array(dataset)

im = dataset.mean(axis=0, keepdims= True)
im = im.reshape(64, 64)
plt.clf()
plt.imshow(im, cmap=plt.cm.gray)
plt.savefig('./faceExpressionDatabase/q1_average.jpg', dpi = 1800)
plt.show()

u, s, v = np.linalg.svd(dataset - dataset.mean(axis=0, keepdims= True))

eigenfaces = v[:9]



plt.clf()
n_row, n_col = 3, 3
for i, face in enumerate(eigenfaces):
    plt.subplot(n_row, n_col, i + 1)
    im = face.reshape(64, 64)
    plt.imshow(im, cmap=plt.cm.gray)



plt.savefig('./faceExpressionDatabase/q1_eigen.jpg', dpi = 1800)
plt.show()



reconstructed_data = []

centered_data = (dataset - dataset.mean(axis=0, keepdims= True))
for face in centered_data:
    
    x_reduced_list = []
    
    for vec in v[:5]:
        x_reduced_list.append(np.dot(face.T, vec))
    
    xhat = dataset.mean(axis=0, keepdims= True)
    for i in range(5):
        xhat += np.dot(x_reduced_list[i], v[i])
    
    
    reconstructed_data.append(xhat)
    """
    im = xhat.reshape(64, 64)
    plt.clf()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()
    """

reconstructed_data = np.array(reconstructed_data).reshape(100,4096)


plt.clf()
n_row, n_col = 10, 10
for i, face in enumerate(dataset):
    plt.subplot(n_row, n_col, i + 1)
    im = face.reshape(64, 64)
    plt.axis('off')
    plt.imshow(im, cmap=plt.cm.gray)

plt.savefig('./faceExpressionDatabase/q2_origin.jpg', dpi = 1800)
plt.show()



plt.clf()
n_row, n_col = 10, 10
for i, face in enumerate(reconstructed_data):
    plt.subplot(n_row, n_col, i + 1)
    im = face.reshape(64, 64)
    plt.axis('off')
    plt.imshow(im, cmap=plt.cm.gray)

plt.savefig('./faceExpressionDatabase/q2_recovered.jpg', dpi = 1800)
plt.show()







ans_k = 0

for n_comp in range(100):
    k = n_comp + 1
    reconstructed_data = []
    centered_data = (dataset - dataset.mean(axis=0, keepdims= True))
    for face in centered_data:
        
        x_reduced_list = []
        
        for vec in v[:k]:
            x_reduced_list.append(np.dot(face.T, vec))
        
        xhat = dataset.mean(axis=0, keepdims= True)
        for i in range(k):
            xhat += np.dot(x_reduced_list[i], v[i])
        
        
        reconstructed_data.append(xhat)
        
    reconstructed_data = np.array(reconstructed_data).flatten()
    
    origin_dataset = dataset.flatten()
    
    rmse = (sum((origin_dataset - reconstructed_data) ** 2)/len(origin_dataset)) ** 0.5
    print(k, rmse, rmse/256)    
    if(rmse/256 < 0.01):
        ans_k = k
        break
    
        




            