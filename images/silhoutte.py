#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:01:28 2021

@author: pablo
"""


from sklearn import metrics
import plotly.graph_objects as go
import numpy as np
from astropy.table import Table
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import re
import shap




'''
Función que se encarga de ver si el parámetro actual mejora la configuración
del algoritmo con los parámetros actuales
'''
def updateMax(puntos, labels, max_score, i, max_sil, max_db, 
              better_clus_number, score_sil_lis, score_db_lis):
    puntuationSil = metrics.silhouette_score(puntos, labels)
    puntuationDB = metrics.davies_bouldin_score(puntos, labels)
    if(puntuationSil - puntuationDB > max_score):
        max_sil = puntuationSil
        max_db = puntuationDB
        max_score = puntuationSil - puntuationDB
        better_clus_number = i
    score_sil_lis.append(puntuationSil)
    score_db_lis.append(puntuationDB)
    
    return max_sil, max_db, max_score, better_clus_number, score_sil_lis, score_db_lis
    


'''
Función encargada de dibujar la tabla que relaciona
el número de clústeres/parámetro de amplitud, con
los valores de shioulette o DB
'''
def plotTable(MAX_ROWS, score_sil_lis, score_db_lis, column_name):
    t = Table([range(2, MAX_ROWS), score_sil_lis, score_db_lis],
                  names=(column_name, 'Silhoutte score', 'DB score'))
    print(t)





'''
Función encargada de devolver los mejores parámetros aplicando el coeficiente
de shiloutte y DB
'''
def scoreFunction(puntos, nombre, cluster, distancia = 0,  **kwargs):
    MAX_CLUSTERS = 8
    MAX_ROWS = 0
    max_score = -99999
    max_sil = -99999
    max_db = -99999
    better_clus_number = 2;
    score_sil_lis = []
    score_db_lis = []
    
    
    if(nombre == "DBSCAN"):
        MAX_ROWS = 5
        for i in range(2, MAX_ROWS):
            dbscan = cluster(eps= (i*distancia), **kwargs).fit(puntos)
            labels = dbscan.labels_
            column_name = 'Epsilon value'
            
            max_sil, max_db, max_score, better_clus_number, score_sil_lis, score_db_lis = updateMax(
                puntos, labels, max_score, i, max_sil, max_db, better_clus_number, score_sil_lis, score_db_lis)
            
    elif(nombre == "OPTICS"):
        MAX_ROWS = 5
        for i in range(2, MAX_ROWS):
            optics = cluster(max_eps = (i*distancia), **kwargs).fit(puntos)
            labels = optics.labels_
            column_name = 'Max epsilon value'
            
            max_sil, max_db, max_score, better_clus_number, score_sil_lis, score_db_lis = updateMax(
                puntos, labels, max_score, i, max_sil, max_db, better_clus_number, score_sil_lis, score_db_lis)
    
    elif(nombre == "MeanShift"):
        MAX_ROWS = 4
        for i in range(2, MAX_ROWS):
            meanShift = cluster(bandwidth= (i*distancia), **kwargs).fit(puntos)
            labels = meanShift.labels_
            column_name = 'Bandwidth'
            max_sil, max_db, max_score, better_clus_number, score_sil_lis, score_db_lis = updateMax(
                puntos, labels, max_score, i, max_sil, max_db, better_clus_number, score_sil_lis, score_db_lis)
            
    
    else:
        MAX_ROWS = MAX_CLUSTERS
        for i in range(2,MAX_CLUSTERS):
            if (nombre == "Agglomerative"):
                agglo = cluster(n_clusters=i, **kwargs).fit(puntos)
                labels = agglo.labels_
                column_name = 'Number Clus'
            elif(nombre == "KMeans"):
                kmeans = cluster(n_clusters=i, **kwargs).fit(puntos)
                labels = kmeans.labels_
                column_name = 'Number Clus'
            elif(nombre == "KMedoids"):
                kmedoids = cluster(n_clusters=i, **kwargs).fit(puntos)
                labels = kmedoids.labels_
                column_name = 'Number Clus'
            
            elif(nombre == "Spectral"):
                spectral = cluster(n_clusters = i, **kwargs).fit(puntos)
                labels = spectral.labels_
                column_name = "Number Clus"
           
                
                    
            max_sil, max_db, max_score, better_clus_number, score_sil_lis, score_db_lis = updateMax(
                puntos, labels, max_score, i, max_sil, max_db, better_clus_number, score_sil_lis, score_db_lis)
    
    
    '''
    En esta parte se detallan algunos datos del resultado
    '''
    if(True):
        plotTable(MAX_ROWS, score_sil_lis, score_db_lis, column_name)
        
 
    return max_sil, max_db, better_clus_number








def f1_score(datos, clusters):
    
    datos = datos.drop(columns = ['Genre'])
    datos = datos.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    clf_kp = LGBMClassifier(colsample_by_tree=0.7)
    cv_scores_kp = cross_val_score(clf_kp, datos, clusters, scoring='f1_weighted')
    print(f'CV F1 score clustering is {np.min(cv_scores_kp)}')
    
    if(len(np.unique(np.array(clusters))) < 3):
        return 0
    else:
        return np.min(cv_scores_kp)
    
    
    
    