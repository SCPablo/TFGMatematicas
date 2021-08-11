#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 20:25:15 2021

@author: pablo
"""

from sklearn.cluster import OPTICS
from silhoutte import scoreFunction
from time import time
from plotResults import plotResultsGrafics, plotResultTable, reduceDimension
import numpy as np 



def eOptics(puntos, distancia):
    print("OPTICS")
    if(distancia == "Juntos"):
        dis = (2)
        xi_val = 0.00000000001
    elif(distancia == "Medio"):
        dis = (4)
        xi_val = 0.001
    else:
        dis = 5
        xi_val = 0.2
    sil_sco, db_sco, eps_val = scoreFunction(
            puntos, "OPTICS", OPTICS, dis, xi=xi_val, min_samples = 3) 
    
    clustering = OPTICS(min_samples=3, xi=xi_val, max_eps = eps_val*dis
                        ).fit(puntos)
    
    t0 = time()
    clustering = clustering.fit(puntos)
    
    print("El valor óptimo del parámetro epsilon es: " + str(eps_val*dis))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    plotResultsGrafics(puntos, clustering, "OPTICS ")
    
    
    
def rOptics1(datos):
    print("OPTICS")
    
    sil_sco, db_sco, xi_val = scoreFunction(
            datos.iloc[:,3:], "OPTICS", OPTICS, 0.05, min_samples = 2) 
    
    clustering = OPTICS(min_samples=1, xi=xi_val*0.05
                        ).fit(datos.iloc[:,3:])
    
    t0 = time()
    clustering = clustering.fit(datos)
    
    print("El valor óptimo del parámetro epsilon es: " + str(0.05*xi_val))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    
    plotResultTable(datos, clustering, "Optics ")
    
    
    
    
def rOptics2(datos):
    print("OPTICS")
    
    sil_sco, db_sco, xi_val = scoreFunction(
            datos, "OPTICS", OPTICS, 0.05, min_samples = 2) 
    
    clustering = OPTICS(min_samples=3, xi=xi_val*0.05
                        ).fit(datos)
    
    t0 = time()
    clustering = clustering.fit(datos)
    
    print("El valor óptimo del parámetro epsilon es: " + str(0.05*xi_val))
    print("La configuración del parámetro anterior produce: " +
          str(len(np.unique(clustering.labels_))) + " clústeres")
    print("En calcular la agrupación ha tardado %.2fs" % ( time() - t0))
    print("Su coeficiente de silhoutte es: " + str(sil_sco))
    print("Su coeficiente de Davies Boulding es: " + str(db_sco))
    print("\n\n\n")
    
    
    reduceDimension(datos, clustering.labels_, "OPTICS")
    