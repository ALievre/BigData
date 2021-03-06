# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 20:58:51 2021

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
from sklearn.neighbors import NearestNeighbors

##########################################################################
#Read data
##################################################################
path = '../artificial/'
databrut = arff.loadarff(open(path+"banana.arff", 'r'))
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

########################################################################
# DBSCAN clustering
########################################################################

#Estimation of epsilon with the nearest neighbors algorithm

n_n = 5
min_s = 5
e = 0.12

nbrs = NearestNeighbors(n_neighbors=n_n).fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)
means = []

for d in distances:
    means.append(d.mean())

plt.figure(3)
plt.plot(range(0, len(means)), sorted(means))
plt.title("Nearest neighbors avec n_neighbors = "+str(n_n))

#Utilisation des métriques pour trouver min_sample

d_sil = []
d_db = []
d_range = range(1,10)
tSilh = []
tDB = []
tMeanSilh = 0.0
tMeanDB = 0.0
metricsName = ['Silhouette', 'Davis-Bouldin']
t_range = 0

dbscan_model = cluster.DBSCAN(eps=e, min_samples=min_s)
dbscan_model.fit(data_scaled)
cl_pred = dbscan_model.labels_

n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
n_noise_ = list(cl_pred).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

for d in d_range:
    dbscan_model = cluster.DBSCAN(eps=e, min_samples=d)
    dbscan_model.fit(data_scaled)
    cl_pred = dbscan_model.labels_
    
    if (n_clusters_ > 1):
        tS1 = time.time()
        silh = metrics.silhouette_score(data_scaled, cl_pred, metric='euclidean')
        tS2 = time.time()
        d_sil.append(silh)
        tDB1 = time.time()
        db = metrics.davies_bouldin_score(data_scaled, cl_pred)
        tDB2 = time.time()
        d_db.append(db)
        
        tSilh.append(tS2-tS1)
        tDB.append(tDB2-tDB1)
        t_range += 1
    else:
        d_sil.append(0)
        d_db.append(0)
        
plt.figure(4)
plt.plot(range(0,len(d_sil)),d_sil)
plt.title("Silhouette score for eps = "+str(e))

plt.figure(5)
plt.plot(range(0,len(d_db)), d_db)
plt.title("D-B score for eps = "+str(e))

dbscan_model = cluster.DBSCAN(eps=e, min_samples=min_s)
dbscan_model.fit(data_scaled)
cl_pred = dbscan_model.labels_

##########################################################################
#Calculation time
##########################################################################
for i in range(0, t_range):
    tMeanSilh += tSilh[i]
    tMeanDB += tDB[i]
tMean = [tMeanSilh/t_range, tMeanDB/t_range]

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
plt.figure()
plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
plt.title("Clustering DBSCAN - Epilson= "+str(e)+" - Minpt= "+str(min_s))
plt.show()