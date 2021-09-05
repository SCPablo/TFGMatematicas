#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:27:07 2021

@author: pablo
"""

from sklearn.cluster import SpectralClustering
from silhoutte import scoreFunction, f1_score
from time import time
from plotResults import plotResultsGrafics, plotResultTable, reduceDimension


def eSpectral(puntos):
    print("SPECTRAL")
    sil_sco, db_sco, n_clus = scoreFunction(
            puntos, "Spectral", SpectralClustering,
        n_init = 10, affinity='nearest_neighbors',
        n_neighbors = 10, eigen_tol = 0.0,
        assign_labels = 'kmeans', degree = 3, coef0 = 1) 
    
    
    t0 = time()
    clustering = SpectralClustering(
        n_clusters = n_clus,
        n_init = 10, affinity='nearest_neighbors',
        n_neighbors = 10, eigen_tol = 0.0,
        assign_labels = 'kmeans', degree = 3, coef0 = 1)
    clustering = clustering.fit(puntos)
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultsGrafics(puntos, clustering, "Spectral ")
    
    

def rSpectral1(datos):
    print("SPECTRAL")
    sil_sco, db_sco, n_clus = scoreFunction(
            datos.iloc[:,3:], "Spectral", SpectralClustering,
        n_init = 10, affinity='nearest_neighbors',
        n_neighbors = 10, eigen_tol = 0.0,
        assign_labels = 'kmeans', degree = 3, coef0 = 1) 
    
    
    t0 = time()
    clustering = SpectralClustering(
        n_clusters = n_clus,
        n_init = 10, affinity='nearest_neighbors',
        n_neighbors = 10, eigen_tol = 0.0,
        assign_labels = 'kmeans', degree = 3, coef0 = 1)
    clustering = clustering.fit(datos.iloc[:,3:])
    
    
    
    valorF = f1_score(datos.iloc[:,2:], clustering.labels_)
    
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultTable(datos, clustering, "Spectral ")
    
    return valorF
    
    
    
    
def rSpectral2(datos):
    print("SPECTRAL")
    sil_sco, db_sco, n_clus = scoreFunction(
            datos, "Spectral", SpectralClustering,
        n_init = 20, affinity='nearest_neighbors',
        n_neighbors = 20, eigen_tol = 0.0,
        assign_labels = 'kmeans', degree = 3, coef0 = 1) 
    
    
    t0 = time()
    clustering = SpectralClustering(
        n_clusters = n_clus,
        n_init = 20, affinity='nearest_neighbors',
        n_neighbors = 20, eigen_tol = 0.0,
        assign_labels = 'kmeans', degree = 3, coef0 = 1)
    clustering = clustering.fit(datos)
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    
    datos['cluster'] = clustering.labels_
    reduceDimension(datos, clustering.labels_, "SPECTRAL")
    