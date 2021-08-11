#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 20:21:21 2021

@author: pablo
"""

from sklearn.cluster import KMeans
from silhoutte import scoreFunction
from time import time
from plotResults import plotResultsGrafics, plotResultTable, reduceDimension


def eKmeans(puntos):
    
    print("KMEANS")
    sil_sco, db_sco, n_clus = scoreFunction(
            puntos, "KMeans", KMeans) 
    clustering = KMeans(
        n_clusters= n_clus)
    t0 = time()
    clustering = clustering.fit(puntos)
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultsGrafics(puntos, clustering, "KMeans ")
    
    
    
    
    
def rKmeans1(datos):
    
    print("KMEANS")
    sil_sco, db_sco, n_clus = scoreFunction(
            datos.iloc[:,3:], "KMeans", KMeans) 
    clustering = KMeans(
        n_clusters= n_clus)
    t0 = time()
    clustering = clustering.fit(datos.iloc[:,3:])
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultTable(datos, clustering, "KMEANS")
    
    
    
    
    
    
def rKmeans2(datos):
    print("KMEANS")
    sil_sco, db_sco, n_clus = scoreFunction(
          datos, "KMeans", KMeans) 
    clustering = KMeans(n_init=20,
        n_clusters= n_clus)
    t0 = time()
    clustering = clustering.fit(datos)
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    
    datos['cluster'] = clustering.labels_
    reduceDimension(datos, clustering.labels_, "KMEANS")
    
    
    