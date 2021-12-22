# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:28:40 2021

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
from sklearn import preprocessing
import scipy.cluster.hierarchy as shc

##########################################################################
#Read data
##################################################################
path = '../artificial/'
databrut = arff.loadarff(open(path+"cure-t2-4k.arff", 'r'))
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

########################################################################
# Preprocessing: standardization of data
########################################################################
scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)


print("-------------------------------------------")
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] 
f1_scaled = data_scaled[:,1] 

plt.figure(2)
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

linkage = ['ward', 'complete', 'average', 'single']

########################################################################
# Dendrogramme
########################################################################
# print("-----------------------------------------")
# print("Dendrogramme données standardisées")

# for l in linkage :
#     distance = shc.linkage(data_scaled, l)
#     plt.figure(figsize=(12, 12))
#     shc.dendrogram(distance,
#                 orientation='top',
#                 distance_sort='descending',
#                 show_leaf_counts=False)
#     plt.title("Dendrogram with linkage = "+str(l))
#     plt.show()

########################################################################
#Agglomerative clustering
########################################################################
print("-----------------------------------------------------------")
print("Appel Aglo Clustering pour une valeur de k fixée")

a_range = range(2,9)
tSilh = []
tDB = []
tMeanSilh = 0.0
tMeanDB = 0.0
metricsName = ['Silhouette', 'Davis-Bouldin']

for l in linkage:
    a_sil = []
    a_db = []
    for a in a_range:
        model_scaled = cluster.AgglomerativeClustering(n_clusters=a, affinity='euclidean', linkage=l)
        model_scaled.fit(data_scaled)
        labels_scaled = model_scaled.labels_
        
        plt.figure(3)
        plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
        plt.title("Données (std) après clustering with linkage = "+str(l))
        plt.show()
    
        tS1 = time.time()
        silh = metrics.silhouette_score(data_scaled, labels_scaled, metric='euclidean')
        tS2 = time.time()
        tDB1 = time.time()
        db = metrics.davies_bouldin_score(data_scaled, labels_scaled)
        tDB2 = time.time()
        a_sil.append(silh)
        a_db.append(db)
        tSilh.append(tS2-tS1)
        tDB.append(tDB2-tDB1)
        
    plt.figure(4)
    plt.plot(range(0,len(a_range)), a_sil, label=l)
    plt.xlabel('Values of A')
    plt.ylabel('Silhouette score')
    plt.title("Silhouette score for a_range = "+str(a_range)+" and linkage = "+str(l))
    plt.legend()
    
    plt.figure(5)
    plt.plot(range(0,len(a_range)), a_db, label=l)
    plt.xlabel('Values of A')
    plt.ylabel('D-B score')
    plt.title("D-B score for a_range = "+str(a_range)+" and linkage = "+str(l))
    plt.legend()

##########################################################################
#Calculation time
##########################################################################
for i in range(0, len(a_range)):
    tMeanSilh += tSilh[i]
    tMeanDB += tDB[i]
tMean = [tMeanSilh/len(a_range), tMeanDB/len(a_range)]

plt.figure(6)
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
a = 2
l = 'ward'
model_scaled = cluster.AgglomerativeClustering(n_clusters=a, affinity='euclidean', linkage=l)
model_scaled.fit(data_scaled)
labels_scaled = model_scaled.labels_

plt.figure(7)
plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
plt.title("Données (std) après clustering with linkage = "+str(l))
plt.show()