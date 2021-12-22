# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:48:47 2021

@author: Agathe Lievre & Assia Nguyen
"""
##########################################################################
#Imports
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pds

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn import preprocessing
import scipy.cluster.hierarchy as shc

##########################################################################
#Read data
##################################################################
#N1
# data=pds.read_csv('./new/n1.csv', sep=',', encoding="windows-1252")
# f0 = data['1'] 
# f1 = data['2']
# datanp = np.array([[f0[x],f1[x]] for x in range(0,2250)])

#N2
# data=pds.read_csv('./new/n2.csv', sep=',', encoding="windows-1252")
# f0 = data['1'] 
# f1 = data['2']
# datanp = np.array([[f0[x],f1[x]] for x in range(0,5500)])

#D32
# data=pds.read_csv('./new/d32.csv', sep=',', encoding="windows-1252")
# f0 = data['1'] # tous les élements de la première colonne
# f1 = data['2'] # tous les éléments de la deuxième colonne
# f2 = data['3'] 
# f3 = data['4'] 
# f4 = data['5'] 
# f5 = data['6'] 
# f6 = data['7'] 
# f7 = data['8'] 
# f8 = data['9'] 
# f9 = data['10'] 
# f10 = data['11'] 
# f11 = data['12'] 
# f12 = data['13'] 
# f13 = data['14'] 
# f14 = data['15'] 
# f15 = data['16'] 
# f16 = data['17'] 
# f17 = data['18'] 
# f18 = data['19'] 
# f19 = data['20'] 
# f20 = data['21'] 
# f21 = data['22'] 
# f22 = data['23'] 
# f23 = data['24'] 
# f24 = data['25'] 
# f25 = data['26'] 
# f26 = data['27'] 
# f27 = data['28'] 
# f28 = data['29'] 
# f29 = data['30'] 
# f30 = data['31'] 
# f31 = data['32'] 
# datanp = np.array([[f0[x],f1[x],f2[x],f3[x],f4[x],f5[x],f6[x],f7[x],f8[x],f9[x],f10[x],f11[x],f12[x],f13[x],f14[x],f15[x],f16[x],f17[x],f18[x],f19[x],f20[x],f21[x],f22[x],f23[x],f24[x],f25[x],f26[x],f27[x],f28[x],f29[x],f30[x],f31[x]] for x in range(0,1024)])

#D64
# data=pds.read_csv('./new/d64.csv', sep=',', encoding="windows-1252")
# f0 = data['1'] # tous les élements de la première colonne
# f1 = data['2'] # tous les éléments de la deuxième colonne
# f2 = data['3'] 
# f3 = data['4'] 
# f4 = data['5'] 
# f5 = data['6'] 
# f6 = data['7'] 
# f7 = data['8'] 
# f8 = data['9'] 
# f9 = data['10'] 
# f10 = data['11'] 
# f11 = data['12'] 
# f12 = data['13'] 
# f13 = data['14'] 
# f14 = data['15'] 
# f15 = data['16'] 
# f16 = data['17'] 
# f17 = data['18'] 
# f18 = data['19'] 
# f19 = data['20'] 
# f20 = data['21'] 
# f21 = data['22'] 
# f22 = data['23'] 
# f23 = data['24'] 
# f24 = data['25'] 
# f25 = data['26'] 
# f26 = data['27'] 
# f27 = data['28'] 
# f28 = data['29'] 
# f29 = data['30'] 
# f30 = data['31'] 
# f31 = data['32'] 
# f32 = data['33'] 
# f33 = data['34']
# f34 = data['35'] 
# f35 = data['36'] 
# f36 = data['37'] 
# f37 = data['38'] 
# f38 = data['39'] 
# f39 = data['40'] 
# f40 = data['41'] 
# f41 = data['42'] 
# f42 = data['43'] 
# f43 = data['44'] 
# f44 = data['45'] 
# f45 = data['46'] 
# f46 = data['47'] 
# f47 = data['48'] 
# f48 = data['49'] 
# f49 = data['50'] 
# f50 = data['51'] 
# f51 = data['52'] 
# f52 = data['53'] 
# f53 = data['54'] 
# f54 = data['55'] 
# f55 = data['56'] 
# f56 = data['57'] 
# f57 = data['58'] 
# f58 = data['59'] 
# f59 = data['60'] 
# f60 = data['61'] 
# f61 = data['62'] 
# f62 = data['63'] 
# f63 = data['64'] 
# datanp = np.array([[f0[x],f1[x],f2[x],f3[x],f4[x],f5[x],f6[x],f7[x],f8[x],f9[x],f10[x],f11[x],f12[x],f13[x],f14[x],f15[x],f16[x],f17[x],f18[x],f19[x],f20[x],f21[x],f22[x],f23[x],f24[x],f25[x],f26[x],f27[x],f28[x],f29[x],f30[x],f31[x],f32[x],f33[x],f34[x],f35[x],f36[x],f37[x],f38[x],f39[x],f40[x],f41[x],f42[x],f43[x],f44[x],f45[x],f46[x],f47[x],f48[x],f49[x],f50[x],f51[x],f52[x],f53[x],f54[x],f55[x],f56[x],f57[x],f58[x],f59[x],f60[x],f61[x],f62[x],f63[x]] for x in range(0,1024)])

#W2
data=pds.read_csv('./new/w2.csv', sep=',', encoding="windows-1252")
datanp = []
arr = data.to_numpy(copy = True)

# Reduce dataframes to numpy arrays
datanp = arr[1::150,:]

f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

##########################################################################
#Affichage des initial data
##################################################################
print("---------------------------------------")
print("Affichage données initiales            ")

plt.figure(1)
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

##########################################################################
#Preprocess
##########################################################################

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

a_range = range(2,7)
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

# #Agglomeratif

# print("-----------------------------------------------------------")
# print("Appel Aglo Clustering 'complete' pour une valeur de k fixée")

# tab_sil = []
# tab_db = []

# for i in range(2,50):
#     tps3 = time.time()
#     model_scaled = cluster.AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
#     model_scaled.fit(data_scaled)
#     #cluster.fit_predict(X)
    
#     tps4 = time.time()
#     labels_scaled = model_scaled.labels_
    
#     # plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
#     # plt.title("Données (std) après clustering")
#     # plt.show()
#     print("nb clusters =",i,", runtime = ", round((tps4 - tps3)*1000,2),"ms")
#     #print("labels", labels)
    
    
#     # Some evaluation metrics
#     silh = metrics.silhouette_score(data_scaled, labels_scaled, metric='euclidean')
#     db = metrics.davies_bouldin_score(data_scaled, labels_scaled)
#     tab_sil.append(silh)
#     tab_db.append(db)
    
# plt.figure()
# plt.plot(range(2,50), tab_sil)
# plt.title("Silhouette score for range_n_clusters = [2:50]")

# plt.figure()
# plt.plot(range(2,50), tab_db)
# plt.title("D-B score for range_n_clusters = [2:50]")

