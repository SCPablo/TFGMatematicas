#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:30:45 2021

@author: pablo
"""

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import itertools
from time import time
import numpy as np
from silhoutte import scoreFunction, f1_score
from plotResults import plotResultsGrafics, plotResultTable, reduceDimension




def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)




def eAgglomerativeClustering(puntos):
    metrics = [ 'average']
    for m in metrics:
        print("AGGLOMERATIVE " + m)
        sil_sco, db_sco, n_clus = scoreFunction(
            puntos, "Agglomerative", AgglomerativeClustering,
            linkage = m, distance_threshold = None) 
        clustering = AgglomerativeClustering(distance_threshold=0,
            n_clusters=None, linkage=m, )
        
        t0 = time()
        
        clustering = clustering.fit(puntos)
        
        print("El número óptimo de clústeres es: " + str(n_clus))
        print("En calcular la agrupación ha tardado %s :\t%.2fs" 
              % (m, time() - t0))
        print("Su coeficiente de silhoutte es: " + str(sil_sco))
        print("Su coeficiente de Davies Boulding es: " + str(db_sco))
        print("\n\n\n")
        plotResultsGrafics(puntos, clustering, "Agglomerative " + m)
        
        #plot_dendrogram(clustering, truncate_mode='level', p=3)
        
        


def rAgglomerativeClustering1(datos):
    
    metrics = ['complete', 'average']
    valorFMax = 0
    for m in metrics:
        print("AGGLOMERATIVE " + m)
        sil_sco, db_sco, n_clus = scoreFunction(
            datos.iloc[:,3:], "Agglomerative", AgglomerativeClustering,
            linkage = m, distance_threshold = None) 
        clustering = AgglomerativeClustering(
            n_clusters= n_clus, linkage=m, )
        t0 = time()
        clustering = clustering.fit(datos.iloc[:,3:])
        
        valorF = f1_score(datos.iloc[:, 2:], clustering.labels_)
        valorFMax = max(valorF, valorFMax)
        
        
        print("El número óptimo de clústeres es: " + str(n_clus))
        print("En calcular la agrupación ha tardado %s :\t%.2fs" 
              % (m, time() - t0))
        print("Su coeficiente de silhoutte es: " + str(sil_sco))
        print("Su coeficiente de Davies Boulding es: " + str(db_sco))
        print("\n\n\n")
        plotResultTable(datos, clustering, "Agglomerative " + m)
        
        
    return valorFMax
        
     
        
def rAgglomerativeClustering2(datos):
    
    metrics = ['complete', 'average']
    for m in metrics:
        print("AGGLOMERATIVE " + m)
        sil_sco, db_sco, n_clus = scoreFunction(
            datos, "Agglomerative", AgglomerativeClustering,
            linkage = m, distance_threshold = None) 
        clustering = AgglomerativeClustering(
            n_clusters= n_clus, linkage=m, )
        t0 = time()
        clustering = clustering.fit(datos)
        
        print("El número óptimo de clústeres es: " + str(n_clus))
        print("En calcular la agrupación ha tardado %s :\t%.2fs" 
              % (m, time() - t0))
        print("Su coeficiente de silhoutte es: " + str(sil_sco))
        print("Su coeficiente de Davies Boulding es: " + str(db_sco))
        print("\n\n\n")
        
        datos['cluster'] = clustering.labels_
        reduceDimension(datos, clustering.labels_, "AGLOMERATIVE " + m)
       
        
        
