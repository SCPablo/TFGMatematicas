#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 20:47:09 2021

@author: pablo
"""

from sklearn.cluster import MeanShift
from silhoutte import scoreFunction, f1_score
from time import time
from plotResults import plotResultsGrafics, plotResultTable, reduceDimension
import numpy as np
from sklearn import metrics


def eMeanShift(puntos, distancia):
    
    print("MEAN SHIFT")
    if(distancia == "Juntos"):
        dis = (1/10)
    elif(distancia == "Medio"):
        dis = (1/2)
    else: dis = 1.3
    sil_sco, db_sco, bandw = scoreFunction(
            puntos, "MeanShift", MeanShift, dis, cluster_all = True) 
    clustering = MeanShift(
        bandwidth= (bandw*dis), cluster_all = True)
    t0 = time()
    clustering = clustering.fit(puntos)
    
    print("El valor óptimo del parámetro amplitud es: " + str(bandw*dis))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultsGrafics(puntos, clustering, "MeanShift ")
    
  
    
def rMeanShift1(datos):
    
    print("MEAN SHIFT")
   
     
    clustering = MeanShift(cluster_all = False)
    t0 = time()
    clustering = clustering.fit(datos.iloc[:,3:])
    sil_sco = metrics.silhouette_score(datos.iloc[:,3:], clustering.labels_)
    db_sco = metrics.davies_bouldin_score(datos.iloc[:,3:], clustering.labels_)
    
    
    
    valorF = f1_score(datos.iloc[:,2:], clustering.labels_)
    
    
    print("El valor óptimo del parámetro amplitud es: " + str(
        clustering.get_params()["bandwidth"]))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultTable(datos, clustering, "MeanShift ")
    
    return valorF
    
    
    
    
def rMeanShift2(datos):
    
    print("MEAN SHIFT")
   
    #sil_sco, db_sco, n_clus = scoreFunction(
     #     datos, "MeanShift", MeanShift, 0.8) 
     
    clustering = MeanShift(bandwidth = 1.5, cluster_all=False)
    t0 = time()
    clustering = clustering.fit(datos)
    sil_sco = metrics.silhouette_score(datos, clustering.labels_)
    db_sco = metrics.davies_bouldin_score(datos, clustering.labels_)
    
    print("El valor óptimo del parámetro amplitud es: " + str(
        clustering.get_params()["bandwidth"]))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    
    datos['cluster'] = clustering.labels_
    reduceDimension(datos, clustering.labels_, "MEANSHIFT")