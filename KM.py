# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:24:54 2019

@author: Hussain
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset= pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimal numbers of cluster
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    Kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0,max_iter=300)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#Applying K_means to the Mall dataset
Kmeans=KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_Kmeans=Kmeans.fit_predict(X)

#visualising the clusters
plt.scatter(X[Y_Kmeans==0,0],X[Y_Kmeans==0,1],s=100,c='red',label='careful')
plt.scatter(X[Y_Kmeans==1,0],X[Y_Kmeans==1,1],s=100,c='blue',label='standard')
plt.scatter(X[Y_Kmeans==2,0],X[Y_Kmeans==2,1],s=100,c='green',label='target')
plt.scatter(X[Y_Kmeans==3,0],X[Y_Kmeans==3,1],s=100,c='cyan',label='careless')
plt.scatter(X[Y_Kmeans==4,0],X[Y_Kmeans==4,1],s=100,c='magenta',label='sensible')
plt.scatter(Kmeans.cluster_centers_[:,0], Kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.title("Clusters of clints")
plt.xlabel("Annual Income K$")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

