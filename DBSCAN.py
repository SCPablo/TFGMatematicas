#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 22:26:53 2021

@author: pablo
"""

from sklearn.cluster import DBSCAN
from silhoutte import scoreFunction
from time import time
from plotResults import plotResultsGrafics, plotResultTable, reduceDimension
import numpy as np 


def eDbscan(puntos, distancia):
    print("DBSCAN")
    if(distancia == "Juntos"):
        dis = (1)
    elif(distancia == "Medio"):
        dis = (1.8)
    else: dis = 3
    sil_sco, db_sco, eps_val = scoreFunction(
            puntos, "DBSCAN", DBSCAN, dis, min_samples = 3) 
    clustering = DBSCAN(
        eps = eps_val*dis, min_samples = 2)
    t0 = time()
    clustering = clustering.fit(puntos)
    
    print("El valor óptimo del parámetro epsilon es: " + str(eps_val*dis))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultsGrafics(puntos, clustering, "DBSCAN ")
    
    
    
    
def rDbscan1(datos):
    print("DBSCAN")
    
    sil_sco, db_sco, eps_val = scoreFunction(
            datos.iloc[:,3:], "DBSCAN", DBSCAN, 0.1, min_samples = 1) 
    clustering = DBSCAN(
        eps = eps_val*0.1, min_samples = 1)
    t0 = time()
    clustering = clustering.fit(datos.iloc[:,3:])
    
    print("El valor óptimo del parámetro epsilon es: " + str(eps_val*0.1))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultTable(datos, clustering, "DBSCAN ")
    
    

def rDbscan2(datos):
    print("DBSCAN")
    
    sil_sco, db_sco, eps_val = scoreFunction(
            datos, "DBSCAN", DBSCAN, 0.25, min_samples = 3) 
    clustering = DBSCAN(
        eps = eps_val*0.25, min_samples = 3)
    t0 = time()
    clustering = clustering.fit(datos)
    
    print("El valor óptimo del parámetro epsilon es: " + str(eps_val*0.5))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    
    reduceDimension(datos, clustering.labels_, "DBSCAN")