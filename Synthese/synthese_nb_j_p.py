# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 01:36:55 2021

@author: Agathe Lievre & Assia Nguyen
"""
##########################################################################
#Imports
##########################################################################
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn import metrics
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

data = pds.read_csv('./new/pluie.csv', sep=',', encoding="windows-1252")
columnsList = data.columns.tolist()
nColumns = data.shape[1] #p
nRows = data.shape[0] #n
index = data.index

##########################################################################
#On garde que les données nb de jours de pluie par mois
##########################################################################
dataNbJPl = data[['JANVIERnb.j.pl', 'FEVRIERnb.j.pl', 'MARSnb.j.pl', 'AVRILnb.j.pl', 'MAInb.j.pl', 'JUINnb.j.pl', 'JUILLETnb.j.pl', 'AOUTnb.j.pl', 'SEPTEMBREnb.j.pl', 'OCTOBREnb.j.pl', 'NOVEMBREnb.j.pl', 'DECEMBREnb.j.pl']]
columnsList = dataNbJPl.columns.tolist()
cityList = data.iloc[:,:1]
cities = data['Ville']

plt.figure(1)

for k in cities :
    indice = index[data['Ville'] == k]
    indiceNb = indice.tolist()
    plt.plot(columnsList, dataNbJPl.iloc[indiceNb[0]], label = k)

plt.tick_params(axis='x', rotation=70)
plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0., ncol=2)
plt.title("Nombre de jours de pluie mensuels moyens")
    
##########################################################################
#On centre et normalise les données
##########################################################################
dataScaled = preprocessing.StandardScaler().fit_transform(dataNbJPl)
print(dataScaled)

dataSN = preprocessing.normalize(dataScaled)
print(dataSN)

plt.figure(2)

for k in cities :
    indice = index[data['Ville'] == k]
    indiceNb = indice.tolist()
    plt.plot(columnsList, dataSN[indiceNb[0],:], label = k)

plt.tick_params(axis='x', rotation=70)
plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0., ncol=2)
plt.title("Données centrées et normalisées")

##########################################################################
#ACP
##########################################################################
acp = PCA(svd_solver = 'full')
data_acp = acp.fit_transform(dataSN)
print(acp.n_components_)
data_acp_names = pds.DataFrame(data_acp ,  columns=['axis1', 'axis2', 'axis3', 'axis4', 'axis5', 'axis6', 'axis7', 'axis8', 'axis9', 'axis10', 'axis11', 'axis12'])
print(data_acp_names)

def display_scree(acp):
    scree = acp.explained_variance_ratio_*100
    print("ACP : recherche d'axe de projection")
    print(scree)
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
plt.figure(3)
display_scree(acp)

#scree plot
plt.figure(4)
plt.plot(np.arange(1,13),acp.explained_variance_)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()

#cumul de variance expliquée
plt.figure(5)
plt.plot(np.arange(1,13),np.cumsum(acp.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel("Factor number")
plt.show()

#seuils pour test des bâtons brisés
bs = 1/np.arange(12,0,-1)
bs = np.cumsum(bs)
bs = bs[::-1]

#test des bâtons brisés
print(pds.DataFrame({'Val.Propre':acp.explained_variance_,'Seuils':bs}))

newData = data_acp_names[['axis1', 'axis2']]
print(newData)

plt.figure(6)
plt.scatter(newData['axis1'], newData['axis2'])
plt.title("Initial data")
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")

##########################################################################
#K-Means clustering
##########################################################################
k_range = range(2,6)
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
tab_sil = []
tab_db = []

for k in k_range:
    kmeanModel = cluster.KMeans(n_clusters=k).fit(newData)
    kmeanModel.fit(newData)
    distortions.append(sum(np.min(cdist(newData, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/newData.shape[0])
    inertias.append(kmeanModel.inertia_)
    mapping1[k] = sum(np.min(cdist(newData, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/newData.shape[0]
    mapping2[k] = kmeanModel.inertia_
    
    clusterK = cluster.KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterK.fit_predict(newData)

    sil_avg = metrics.silhouette_score(newData, cluster_labels)
    db_avg = metrics.davies_bouldin_score(newData, cluster_labels)
    tab_sil.append(sil_avg)
    tab_db.append(db_avg)
    
plt.figure(7)
plt.plot(k_range, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distorsion')
plt.title("Elbow method for k_range = "+str(k_range))
plt.show()

plt.figure(8)
plt.plot(k_range, tab_sil)
plt.title("Silhouette score for k_range = "+str(k_range))

plt.figure(9)
plt.plot(k_range, tab_db)
plt.title("D-B score for k_range = "+str(k_range))

model_km = cluster.KMeans(n_clusters=3, init='k-means++',n_init=50)
model_km.fit(newData)
labels_km = model_km.labels_

# Résultat du clustering
plt.figure(10)
plt.scatter(newData['axis1'], newData['axis2'], c=labels_km, s=8)
plt.title("Données  après K-Means clustering")
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")

##########################################################################
#Agglomerative clustering
##########################################################################
linkage = ["complete","ward","average","single"]
a_range = range(2,6)

for l in linkage :
    
    a_sil = []
    a_db = []
    
    for a in a_range :
        aggloModel = cluster.AgglomerativeClustering(n_clusters=a, affinity='euclidean', linkage = l)
        aggloModel.fit(newData)
        labels_agglo = aggloModel.labels_
        
        silh = metrics.silhouette_score(newData, labels_agglo, metric='euclidean')
        db = metrics.davies_bouldin_score(newData, labels_agglo)
        a_sil.append(silh)
        a_db.append(db)
    
    plt.figure(11)
    plt.plot(a_range, a_sil, label=l)
    plt.xlabel('Values of A')
    plt.ylabel('Silhouette score')
    plt.title("Silhouette score for a_range = "+str(a_range))
    plt.legend()

    plt.figure(12)
    plt.plot(a_range, a_db, label=l)
    plt.xlabel('Values of A')
    plt.ylabel('D-B score')
    plt.title("D-B score for a_range = "+str(a_range))
    plt.legend()
    
new_a_sil = []
new_a_db = []    

for a in a_range :
    aggloModel = cluster.AgglomerativeClustering(n_clusters=a, affinity='euclidean', linkage = 'ward')
    aggloModel.fit(newData)
    labels_agglo = aggloModel.labels_
    
    silh = metrics.silhouette_score(newData, labels_agglo, metric='euclidean')
    db = metrics.davies_bouldin_score(newData, labels_agglo)
    new_a_sil.append(silh)
    new_a_db.append(db)

plt.figure(13)
plt.plot(a_range, new_a_sil, label='ward')
plt.xlabel('Values of A')
plt.ylabel('Silhouette score')
plt.title("Silhouette score for a_range = "+str(a_range))
plt.legend()

plt.figure(14)
plt.plot(a_range, new_a_db, label='ward')
plt.xlabel('Values of A')
plt.ylabel('D-B score')
plt.title("D-B score for a_range = "+str(a_range))
plt.legend()

aggloModel = cluster.AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage = 'ward')
aggloModel.fit(newData)
labels_agglo = aggloModel.labels_

# Résultat du clustering
plt.figure(15)
plt.scatter(newData['axis1'], newData['axis2'], c=labels_agglo, s=8)
plt.title("Données  après Agglomerative clustering")
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")

##########################################################################
#DBSCAN clustering
##########################################################################
nbNN = 10
minSamples = 3
epsilon = 0.28
d_range = range(1,10)

nbrs = NearestNeighbors(n_neighbors=nbNN).fit(newData)
distances, indices = nbrs.kneighbors(newData)
means = []

for i in distances:
    means.append(i.mean())

plt.figure(16)
plt.plot(range(0, len(means)), sorted(means))
plt.title("Nearest neighbors avec n_neighbors = "+str(nbNN))
plt.xlabel("Number of nearest neighbors")
plt.ylabel("Epsilon")

d_sil = []
d_db = []

dbscan_model = cluster.DBSCAN(eps=epsilon, min_samples=minSamples)
dbscan_model.fit(newData)
labels_dbscan = dbscan_model.labels_

n_clusters_ = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_ = list(labels_dbscan).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

for d in d_range:
    dbscan_model = cluster.DBSCAN(eps=epsilon, min_samples=d)
    dbscan_model.fit(newData)
    labels_dbscan = dbscan_model.labels_
    
    n_clusters_ = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise_ = list(labels_dbscan).count(-1)
    
    if (n_clusters_ > 1):
        silh = metrics.silhouette_score(newData, labels_dbscan, metric='euclidean')
        d_sil.append(silh)
        db = metrics.davies_bouldin_score(newData, labels_dbscan)
        d_db.append(db)
    else:
        d_sil.append(0)
        d_db.append(0)
        
plt.figure(17)
plt.plot(range(0,len(d_sil)), d_sil)
plt.title("Silhouette score for eps = "+str(epsilon))

plt.figure(18)
plt.plot(range(0,len(d_db)), d_db)
plt.title("D-B score for eps = "+str(epsilon))

dbscan_model = cluster.DBSCAN(eps=epsilon, min_samples=minSamples)
dbscan_model.fit(newData)
labels_dbscan = dbscan_model.labels_

#Résultat du clustering
plt.figure(19)
plt.scatter(newData['axis1'], newData['axis2'], c=labels_dbscan, s=8)
plt.title("Données  après DBSCAN clustering")
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")

##########################################################################
#Projection des données
##########################################################################

#nuage
plt.figure(20)
ax1 = newData['axis1']
ax2 = newData['axis2']
plt.scatter(newData['axis1'], newData['axis2'], s=10)
plt.title("Nuage des données")
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")
plt.axvline(x=0, c='grey', ls='--', linewidth=1)
plt.axhline(y=0, c='grey', ls='--', linewidth=1)

for i, name in enumerate(cities) :
    plt.annotate(name, (ax1[i], ax2[i]), xytext=(ax1[i]+0.05, ax2[i]-0.01), size=8)

#cercle des corrélations
plt.figure(20)
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
plt.title("Cercle des corrélations")
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")
plt.plot([-1,1],[0,0],color='grey',linestyle='--',linewidth=1)
plt.plot([0,0],[-1,1],color='grey',linestyle='--',linewidth=1)

for i, name in enumerate(cities) :
    plt.annotate(name, (ax1[i], ax2[i]), xytext=(ax1[i], ax2[i]), size=8)
      
cercle = plt.Circle((0,0),1,color='blue',fill=False)
ax.add_patch(cercle)
