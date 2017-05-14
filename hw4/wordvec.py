# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:51:55 2017

@author: HSIN
"""


"""
%load_ext autoreload
%autoreload 2
%reload_ext word2vec
%matplotlib inline

"""

import word2vec
import numpy as np
from sklearn.manifold import TSNE
import nltk
import matplotlib.pyplot as plt
from adjustText import adjust_text


w2v_size = 1000
k = 750


word2vec.word2phrase('./Book5TheOrderOfThePhoenix/all.txt', './Book5TheOrderOfThePhoenix/all-phrases', verbose=True)
word2vec.word2vec('./Book5TheOrderOfThePhoenix/all-phrases', './Book5TheOrderOfThePhoenix/all.bin', size=w2v_size, verbose=True)


w2v_model = word2vec.load('./Book5TheOrderOfThePhoenix/all.bin')


"""
word2vec.word2clusters('./Book5TheOrderOfThePhoenix/all.txt', './Book5TheOrderOfThePhoenix/all-clusters.txt', w2v_size, verbose=True)
w2v_model.clusters = word2vec.load_clusters('./Book5TheOrderOfThePhoenix/all-clusters.txt')
"""



tsne_model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vocab_tsne = tsne_model.fit_transform(w2v_model.vectors[:k]) 

tag_list = [item[1] for item in nltk.pos_tag(w2v_model.vocab[:k])]

selected_vocab = []
selected_tsne = []


punc = ", . : ; ! ? ' ’ ‘ ” “ \" _ - "

for i, tag in enumerate(tag_list):
    if tag in ['JJ','NNP','NN','NNS']:
        if not any((c in punc for c in w2v_model.vocab[i])):
            selected_vocab.append(w2v_model.vocab[i])
            selected_tsne.append(vocab_tsne[i])
        


selected_vocab = np.asarray(selected_vocab)
selected_tsne = np.asarray(selected_tsne)
        





"""
plt.clf()
plt.scatter(selected_tsne[:, 0], selected_tsne[:, 1])
for label, x, y in zip(selected_vocab, selected_tsne[:, 0], selected_tsne[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')


plt.savefig('./Book5TheOrderOfThePhoenix/w2v.png')
        
plt.show()
"""



def plot_scatter(adjust, xvalue, yvalue, label):
    plt.clf()
    plt.figure(figsize=(16, 9))
    plt.scatter(xvalue, yvalue, s=15, c=np.random.rand(len(label),1), edgecolors='None', alpha=0.5)
    texts = []
    for x, y, s in zip(xvalue, yvalue, label):
        texts.append(plt.text(x, y, s, size=7))
    if adjust:
        plt.title(str(adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5)))+' iterations')
    
    
    plt.savefig('./Book5TheOrderOfThePhoenix/w2v.png', dpi = 1800)
    plt.show()
        
plot_scatter(adjust = True, xvalue=selected_tsne[:, 0], yvalue=selected_tsne[:, 1], label=selected_vocab)