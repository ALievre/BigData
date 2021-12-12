# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:08:23 2021

@author: huguet
@modifications: A. Lievre, A. Nguyen
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn import preprocessing

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
#    2d-4c-no9.arff

path = '../artificial/'
databrut = arff.loadarff(open(path+"banana.arff", 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])
#print(databrut)
#print(datanp)

##################################################################
# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            ")
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.figure(1)
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

########################################################################
# AUTRES VISUALISATION DU JEU DE DONNEES
# (histogrammes par exemple,)
# But : essayer d'autres types de plot 
#N_points = 100000

########################################################################
n_bins = 100

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

#histogram
# We can set the number of bins with the *bins* keyword argument.
plt.figure(2)
axs[0].hist(f0, bins=n_bins) #axe ordonnée
axs[1].hist(f1, bins=n_bins) #axe abscisse

#2D hexagonal binning 
plt.figure(3);
plt.hexbin(f0,f1,gridsize=40);

#2D histogram
plt.figure(4);
plt.hist2d(f0,f1,100);

########################################################################
# STANDARDISER ET VISUALISER 
# But : comparer des méthodes de standardisation, ...
########################################################################
## Standardisation (base) -> centrer en 0
scaler=preprocessing.StandardScaler().fit_transform(datanp);
plt.figure(5);
plt.scatter(scaler[:,0], scaler[:,1], s=8);
plt.title("Standardisation");
plt.show();

#MinMaxScaler
min_max_scaler=preprocessing.MinMaxScaler();
X_scaler=min_max_scaler.fit_transform(datanp);
plt.figure(6);
plt.scatter(X_scaler[:,0], X_scaler[:,1], s=8);
plt.title("Min max std");
plt.show();
