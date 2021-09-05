#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 20:39:29 2021

@author: pablo
"""

from sklearn_extra.cluster import KMedoids
from silhoutte import scoreFunction, f1_score
from time import time
from plotResults import plotResultsGrafics, plotResultTable, reduceDimension




def eKmedoids(puntos):
    
    print("KMEDOIDS")
    sil_sco, db_sco, n_clus = scoreFunction(
            puntos, "KMedoids", KMedoids, init='k-medoids++') 
    clustering = KMedoids(
        n_clusters= n_clus, init='k-medoids++')
    t0 = time()
    clustering = clustering.fit(puntos)
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultsGrafics(puntos, clustering, "KMedoids ")
    
    
    

def rKmedoids1(datos):
    
    print("KMEDOIDS")
    sil_sco, db_sco, n_clus = scoreFunction(
            datos.iloc[:,3:], "KMedoids", KMedoids, init='k-medoids++') 
    clustering = KMedoids(
        n_clusters= n_clus, init='k-medoids++')
    t0 = time()
    clustering = clustering.fit(datos.iloc[:, 3:])
    
    valorF = f1_score(datos.iloc[:,2:], clustering.labels_)
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    
    
    plotResultTable(datos, clustering, "KMEDOIDS")
    
    return valorF
    
    
    
def rKmedoids2(datos):
    
    print("KMEDOIDS")
    sil_sco, db_sco, n_clus = scoreFunction(
            datos, "KMedoids", KMedoids, init='k-medoids++') 
    clustering = KMedoids(
        n_clusters= n_clus, init='k-medoids++')
    t0 = time()
    clustering = clustering.fit(datos)
    
    print("El número óptimo de puntos es: " + str(n_clus))
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    datos['cluster'] = clustering.labels_
    reduceDimension(datos, clustering.labels_, "KMEDOIDS")
    