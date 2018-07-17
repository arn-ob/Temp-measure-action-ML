# -*- coding: utf-8 -*-

# import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Import dataset
#datasettemp = pd.read_csv('dataset/datatemp.csv')
datasethum = pd.read_csv('dataset/tempData.csv')
X = datasethum.iloc[:, [0, 1]].values



# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of Temp respect to Time')
plt.xlabel('Time')
plt.ylabel('TEMP')
plt.legend()
plt.show()

