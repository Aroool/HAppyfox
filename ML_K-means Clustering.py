# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
     

data = pd.read_csv('D:/Edu_Exa_Scu/Dataset.csv')
data.drop(['Timestamp'], axis = 1, inplace = True)
data.head()
     
new_data = data.dropna(axis = 0)
     

new_data.shape

new_data.info()

data1 = new_data.copy()
     

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
     

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data1 = new_data.apply(LabelEncoder().fit_transform)
data1


from sklearn.decomposition import PCA
     

pca = PCA()
compressed_data = pca.fit_transform(data1)
     

pca.explained_variance_ratio_  

plt.figure(figsize=(10,8))
plt.plot(range(1,21), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title("Explained variance by components")
plt.xlabel('Number of components')
plt.ylabel('Cumulative Explained Variance')

pca = PCA(n_components = 3)
     

pca.fit(data1)
     

pca.transform(data1)

score_pca = pca.transform(data1)

wcss = []    #WCSS is the sum of squared distance between each point and the centroid in a cluster

for k in range(1,9):
  kmeans = KMeans(n_clusters = k)
  kmeans.fit(score_pca)
  wcss_iter = kmeans.inertia_
  wcss.append(wcss_iter)

wcss

plt.figure(figsize=(10,8))
plt.plot(range(1,9), wcss, marker = 'o', linestyle = '-')
plt.xlabel('Number of cluster (K)')
plt.ylabel('wcss')
plt.title('K-means with PCA clustering')
     

km  = KMeans(n_clusters= 3)
km

km.fit(score_pca)
     

y_pred = km.fit_predict(score_pca)
y_pred

df = pd.concat([data1.reset_index(drop = True), pd.DataFrame(score_pca)], axis = 1)
df.columns.values[-3: ] = ['Component 1', 'Component 2', 'Component 3']
df['Segment K-means PCA'] = km.labels_
     

df.head()

df['segment'] = df['Segment K-means PCA'].map({0:'first', 1:'second', 2:'third'})
     

df.head()
