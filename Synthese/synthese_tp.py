# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:54:56 2021

@author: Agathe Lievre & Assia Nguyen
"""
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors



data=pds.read_csv('./new/n1.csv', sep=',', encoding="windows-1252")

f0 = data['1'] # tous les élements de la première colonne
f1 = data['2'] # tous les éléments de la deuxième colonne

datanp = np.array([[f0[x],f1[x]] for x in range(0,2250)])

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

#K-Means
k=3
range_n_clusters = [2, 3, 4, 5, 6]

model_km = cluster.KMeans(n_clusters=k, init='k-means++',n_init=50)
#model_km = cluster.KMeans(init='random',random_state=150,n_init=100)
model_km.fit(datanp)
labels_km = model_km.labels_

# Nb iteration of this method
iteration = model_km.n_iter_

# Résultat du clustering
plt.scatter(f0, f1, c=labels_km, s=8)
plt.title("Données (init) après clustering")
plt.show()
#print("nb clusters =",k,", nb iter =",iteration, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels_km)

#Méthode du coude
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}

tps1_elbow = time.time()

for k in range_n_clusters:
    kmeanModel = cluster.KMeans(n_clusters=k).fit(datanp)
    kmeanModel.fit(datanp)
    distortions.append(sum(np.min(cdist(datanp, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/datanp.shape[0])
    inertias.append(kmeanModel.inertia_)
    mapping1[k] = sum(np.min(cdist(datanp, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/datanp.shape[0]
    mapping2[k] = kmeanModel.inertia_
    
tps2_elbow = time.time()
print("runtime = ", round((tps2_elbow - tps1_elbow)*1000,2),"ms")
    
plt.figure(7)
plt.plot(range(2,7), distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distorsion')
plt.title("Elbow method for range_n_clusters = [2, 3, 4, 5, 6]")
plt.show()

#Silhouette and D-B score
tps1_sil = time.time()
tab_sil = []
tab_db = []

for n_clusters in range_n_clusters:
    
    clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(datanp)

    sil_avg = metrics.silhouette_score(datanp, cluster_labels)
    db_avg = metrics.davies_bouldin_score(datanp, cluster_labels)
    tab_sil.append(sil_avg)
    tab_db.append(db_avg)
    
tps2_sil = time.time()
print("runtime = ", round((tps2_sil - tps1_sil)*1000,2),"ms")

plt.figure()
plt.plot(range(2,7), tab_sil)
plt.title("Silhouette score for range_n_clusters = [2, 3, 4, 5, 6]")

plt.figure()
plt.plot(range(2,7), tab_db)
plt.title("D-B score for range_n_clusters = [2, 3, 4, 5, 6]")

#Agglomeratif
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

print("-----------------------------------------------------------")
print("Appel Aglo Clustering 'complete' pour une valeur de k fixée")

tab_sil = []
tab_db = []

for i in [2,3,4,5,6]:
    tps3 = time.time()
    k=i
    model_scaled = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    model_scaled.fit(data_scaled)
    #cluster.fit_predict(X)
    
    tps4 = time.time()
    labels_scaled = model_scaled.labels_
    
    plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
    plt.title("Données (std) après clustering")
    plt.show()
    print("nb clusters =",k,", runtime = ", round((tps4 - tps3)*1000,2),"ms")
    #print("labels", labels)
    
    
    # Some evaluation metrics
    silh = metrics.silhouette_score(data_scaled, labels_scaled, metric='euclidean')
    db = metrics.davies_bouldin_score(data_scaled, labels_scaled)
    tab_sil.append(silh)
    tab_db.append(db)
    
plt.figure()
plt.plot(range(2,7), tab_sil)
plt.title("Silhouette score for range_n_clusters = [2, 3, 4, 5, 6]")

plt.figure()
plt.plot(range(2,7), tab_db)
plt.title("D-B score for range_n_clusters = [2, 3, 4, 5, 6]")


#DBSCAN
n_n = 50
min_s = 50
e = 0.25

nbrs = NearestNeighbors(n_neighbors=n_n).fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)
means = []

for i in distances:
    means.append(i.mean())

plt.plot(range(0, len(means)), sorted(means))
plt.title("Nearest neighbors avec n_neighbors = "+str(n_n))

tab_sil = []
tab_db = []

dbscan_model = cluster.DBSCAN(eps=e, min_samples=min_s)
dbscan_model.fit(data_scaled)
cl_pred = dbscan_model.labels_

n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
n_noise_ = list(cl_pred).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

for i in range(1,100):
    dbscan_model = cluster.DBSCAN(eps=e, min_samples=i)
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
plt.title("Silhouette score for eps = "+str(e))

plt.figure()
plt.plot(range(0,len(tab_db)), tab_db)
plt.title("D-B score for eps = "+str(e))

dbscan_model = cluster.DBSCAN(eps=e, min_samples=min_s)
dbscan_model.fit(data_scaled)
cl_pred = dbscan_model.labels_

#Plot results
plt.figure()
plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
plt.title("Clustering DBSCAN - Epilson= "+str(e)+" - Minpt= "+str(min_s))
plt.show()