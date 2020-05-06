#!/usr/bin/env python
# coding: utf-8

# In[2]:


import spacy 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from spacy import displacy
from spacy.matcher import Matcher 
from spacy.tokens import Span 
# from spacy.util import filter_spans
from spacy.lang.en.stop_words import STOP_WORDS
import string
import neuralcoref

from sklearn.cluster import KMeans

nlp = spacy.load('en_core_web_lg')

# this line makes any plots display in the notebook
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


aspects = []
with open("my_aspectsCOREF.txt") as file:
    for line in file:
        aspects.append(line.split("'")[1])

# aspects = aspects[0:200] 
    

wordvecs = np.array([[0.0, 0.0] for i in range(len(aspects))])
with open("wordvecs.txt") as file:
    i = 0
    for line in file:
        if(i >= len(wordvecs)):
            break
        
        line = line[1:-2]
        split_line = (line.split())
        
        wordvecs[i][0] = float(split_line[0])
        wordvecs[i][1] = float(split_line[1])
    
        i += 1


# In[11]:


# # intialise pca model and tell it to project data down onto 2 dimensions
# pca = PCA(n_components = 2)


def get_word_vectors(words):
    return([nlp(word).vector for word in words])

word_vectors = get_word_vectors(aspects)  
# print("vectros obtained")

# with open('wordbeddings.txt', 'a') as f:
#     for i in range(len(word_vectors)):
#         print(word_vectors[i], file = f)

kmeans = KMeans(n_clusters = 5)
kmeans.fit(word_vectors)
y_kmeans = kmeans.predict(word_vectors)
# centers = kmeans.cluster_centers_


# pca.fit(word_vectors)
# word_vecs_2d = pca.transform(word_vectors)
# pca = PCA(n_components = 2)
# pca.fit(centers)
# centers = pca.transform(centers)

# print("pca done")

# # Tell our (fitted) pca model to transform our 300D data down onto 2D using the 
# # instructions it learnt during the fit phase.
# word_vecs_2d = pca.transform(word_vectors)
# centers = pca.transform(centers)
# print("centers", centers)

# with open('wordvecs.txt', 'a') as f:
#     for i in range(len(word_vecs_2d)):
#         print(word_vecs_2d[i], file = f)

kmeans2 = KMeans(n_clusters = 5)
kmeans2.fit(wordvecs)
y_kmeans2 = kmeans2.predict(wordvecs)

# create a nice big plot 
plt.figure(figsize=(20,15))

plt.scatter(wordvecs[:, 0], wordvecs[:, 1], c = y_kmeans, s = 15, cmap = 'viridis', alpha = 0.2)
# plt.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1], c = y_kmeans, s = 15, cmap = 'viridis', alpha = 0.5)
centers = kmeans2.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s = 100, alpha = 0.8)

# for each word and coordinate pair: draw the text on the plot
# for word, coord in zip(aspects, wordvecs):
#     x, y = coord
#     plt.text(x, y, word, size= 8)

# show the plot
plt.show()