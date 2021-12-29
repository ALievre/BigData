# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:08:23 2021

@authors: A. Lievre, A. Nguyen
"""
##########################################################################
#Imports
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.spatial.distance import cdist

##########################################################################
#Read data
##################################################################
path = '../artificial/'
databrut = arff.loadarff(open(path+"dartboard1.arff", 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

##########################################################################
#Affichage des initial data
##################################################################
print("---------------------------------------")
print("Affichage données initiales            ")
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

plt.figure(1)
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

##########################################################################
#K-Means clustering
##################################################################
metricsName = ['Elbow', 'Silhouette', 'Davis-Bouldin']

k_range = range(2,11)
k_clusters = 4
k_distortions = []
k_sil = []
k_db = []
tElbow = []
tSilh = []
tDB = []
tMeanElbow = 0.0
tMeanSilh = 0.0
tMeanDB = 0.0

for k in k_range:
    tE1 = time.time()
    kmeanModel = cluster.KMeans(n_clusters=k).fit(datanp)
    kmeanModel.fit(datanp)
    k_distortions.append(sum(np.min(cdist(datanp, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/datanp.shape[0])
    tE2 = time.time()
    
    clusterK = cluster.KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterK.fit_predict(datanp)
    
    tS1 = time.time()
    sil_avg = metrics.silhouette_score(datanp, cluster_labels)
    tS2 = time.time()
    
    tDB1 = time.time()
    db_avg = metrics.davies_bouldin_score(datanp, cluster_labels)
    tDB2 = time.time()
    
    k_sil.append(sil_avg)
    k_db.append(db_avg)
    tElbow.append(tE2-tE1)
    tSilh.append(tS2-tS1)
    tDB.append(tDB2-tDB1)
    
##########################################################################
#Elbow method
##########################################################################
plt.figure(2)
plt.plot(k_range, k_distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distorsion')
plt.title("Elbow method for k_range = "+str(k_range))
plt.show()

##########################################################################
#Silhouette score
##########################################################################
plt.figure(3)
plt.plot(k_range, k_sil)
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title("Silhouette score for k_range = "+str(k_range))

##########################################################################
#DB score
##########################################################################
plt.figure(4)
plt.plot(k_range, k_db)
plt.xlabel('Values of K')
plt.ylabel('D-B score')
plt.title("D-B score for k_range = "+str(k_range))

##########################################################################
#Calculation time
##########################################################################
for i in range(0, len(k_range)):
    tMeanElbow += tElbow[i]
    tMeanSilh += tSilh[i]
    tMeanDB += tDB[i]
tMean = [tMeanElbow/len(k_range), tMeanSilh/len(k_range), tMeanDB/len(k_range)]

plt.figure(5)
x = np.arange(len(metricsName))
width = 0.4
fig, ax = plt.subplots()
rect = ax.bar(x, tMean, width)
ax.set_ylabel('Temps en secondes')
ax.set_title('Temps de calcul moyen des métriques')
plt.xticks(x, metricsName)
ax.bar_label(rect, padding=1)
fig.tight_layout()
plt.show()

##########################################################################
#Résultat du clustering
##########################################################################
model_km = cluster.KMeans(n_clusters=k_clusters, init='k-means++',n_init=50)
model_km.fit(datanp)
labels_km = model_km.labels_
 
plt.figure(6)
plt.scatter(f0, f1, c=labels_km, s=8)
plt.title("Données  après K-Means clustering")
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")