# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 20:58:51 2021

@author: huguet


"""

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features

# Note 1 : 
# dans les jeux de données considérés : 2 features (dimension 2 seulement)
# t =np.array([[1,2], [3,4], [5,6], [7,8]]) 
#
# Note 2 : 
# le jeu de données contient aussi un numéro de cluster pour chaque point
# --> IGNORER CETTE INFORMATION ....
#    2d-4c-no9.arff   xclara.arff
#  2d-4c-no4    spherical_4_3 
# cluto-t8-8k  cluto-t4-8k cluto-t5-8k cluto-t7-8k diamond9 banana
path = '../artificial/'
databrut = arff.loadarff(open(path+"banana.arff", 'r'))
data = [[x[0],x[1]] for x in databrut[0]]
datanp = np.array([[x[0],x[1]] for x in databrut[0]])


########################################################################
# Preprocessing: standardization of data
########################################################################

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(datanp)

data_scaled = scaler.transform(datanp)

import scipy.cluster.hierarchy as shc

print("-------------------------------------------")
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

########################################################################
# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
#  
# print("-----------------------------------------------------------")
# print("DBSCAN - Eps=3, MinPts=5")
# distance=3
# min_pts=5
# dbscan_model = cluster.DBSCAN(eps=distance, min_samples=min_pts)
# dbscan_model.fit(data)

# cl_pred = dbscan_model.labels_
# #cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data)

# # Plot results
# plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
# plt.title("Clustering DBSCAN - Epilson=3 - Minpt=5")
# plt.show()
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
# n_noise_ = list(cl_pred).count(-1)
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)

# Some evaluation metrics
#silh = metrics.silhouette_score(data_scaled, cl_pred, metric='euclidean')
#print("Coefficient de silhouette : ", silh)
#db = metrics.davies_bouldin_score(data_scaled, cl_pred)
#print("Coefficient de Davies Bouldin : ", db)

# Another example
# print("-----------------------------------------------------------")
# print("DBSCAN - Eps=0.01, MinPts=3")
# distance=0.01
# min_pts=3
# dbscan_model = cluster.DBSCAN(eps=distance, min_samples=min_pts)
# dbscan_model.fit(data)

# cl_pred = dbscan_model.labels_
# # Plot results
# plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
# plt.title("Clustering DBSCAN - Epilson=0.02 - Minpt=5")
# plt.show()
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
# n_noise_ = list(cl_pred).count(-1)
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)

# # Some evaluation metrics
# silh = metrics.silhouette_score(data_scaled, cl_pred, metric='euclidean')
# print("Coefficient de silhouette : ", silh)
# db = metrics.davies_bouldin_score(data_scaled, cl_pred)
# print("Coefficient de Davies Bouldin : ", db)

# # Another example
# print("-----------------------------------------------------------")
# print("DBSCAN - Eps=0.04, MinPts=5")
# distance=0.04
# min_pts=5
# dbscan_model = cluster.DBSCAN(eps=distance, min_samples=min_pts)
# dbscan_model.fit(data)

# cl_pred = dbscan_model.labels_
# # Plot results
# plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
# plt.title("Clustering DBSCAN - Epilson=0.04 - Minpt=5")
# plt.show()
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
# n_noise_ = list(cl_pred).count(-1)
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)

# # Some evaluation metrics
# silh = metrics.silhouette_score(data_scaled, cl_pred, metric='euclidean')
# print("Coefficient de silhouette : ", silh)
# db = metrics.davies_bouldin_score(data_scaled, cl_pred)
# print("Coefficient de Davies Bouldin : ", db)

########################################################################
# FIND "interesting" values of epsilon and min_samples 
# using distances of the k NearestNeighbors for each point of the dataset
#
# Note : a point x is considered to belong to its own neighborhood  

nbrs = NearestNeighbors(n_neighbors=5).fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)
means = []

for i in distances:
    means.append(i.mean())

plt.plot(range(0, len(means)), sorted(means))
plt.title("Nearest neighbors avec n_neighbors = 5")

tab_sil = []
tab_db = []

dbscan_model = cluster.DBSCAN(eps=0.04, min_samples=1)
dbscan_model.fit(data)
cl_pred = dbscan_model.labels_

n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
n_noise_ = list(cl_pred).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

#Plot results
# plt.figure()
# plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
# plt.title("Clustering DBSCAN - Epilson=0.04 - Minpt=5")
# plt.show()

for i in range(1,50):
    dbscan_model = cluster.DBSCAN(eps=0.04, min_samples=i)
    dbscan_model.fit(data_scaled)
    cl_pred = dbscan_model.labels_
    
    n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
    #print('Estimated number of clusters: %d' % n_clusters_)
    n_noise_ = list(cl_pred).count(-1)
    
    if (n_clusters_ > 1):
        # Some evaluation metrics
        silh = metrics.silhouette_score(data_scaled, cl_pred, metric='euclidean')
        tab_sil.append(silh)
        db = metrics.davies_bouldin_score(data_scaled, cl_pred)
        tab_db.append(db)
    else:
        tab_sil.append(0)
        tab_db.append(0)
        
plt.figure()
plt.plot(range(0,len(tab_sil)), tab_sil)
plt.title("Silhouette score for eps = 0,04")

plt.figure()
plt.plot(range(0,len(tab_db)), tab_db)
plt.title("D-B score for eps = 0,04")