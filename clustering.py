#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 12:20:30 2021

@author: pablo
"""




from agglomerative import eAgglomerativeClustering, rAgglomerativeClustering1, rAgglomerativeClustering2
from KMeans import eKmeans, rKmeans1, rKmeans2
from KMedoids import eKmedoids, rKmedoids1, rKmedoids2
from MeanShift import eMeanShift, rMeanShift1, rMeanShift2
from DBSCAN import eDbscan, rDbscan1, rDbscan2
from OPTICS import eOptics, rOptics1, rOptics2
from Espectral import eSpectral, rSpectral1, rSpectral2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from plotResults import drawHeatMap





'''
Función encargada de llamar a todos los algoritmos de clustering
'''
def experimentClustering(puntos, distancia):
    
    eAgglomerativeClustering(puntos)
    eKmeans(puntos)
    eKmedoids(puntos)
    eMeanShift(puntos, distancia)
    eDbscan(puntos, distancia)
    eOptics(puntos, distancia)
    eSpectral(puntos)
    
    
    
    
'''    
Función encargada de escalar las columnas numéricas entre 0 y 100
'''
def preprocessSongs(df):
    
    scaler = MinMaxScaler()
    transformed_df = scaler.fit_transform(df.iloc[:, 3:])
    transformed_df = pd.DataFrame(transformed_df)
    transformed_df["TrackName"] = df["TrackName"]
    transformed_df["Artist"] = df["ArtistName"]
    transformed_df["Genre"] = df["Genre"]
    
    cols = transformed_df.columns.tolist()
    for i in range(0,3):
        cols = cols[-1:] + cols[:-1]
    transformed_df = transformed_df[cols]
    
    transformed_df.columns = df.columns
    
    
    return transformed_df
    
    
    
def preprocessCorona(df):
    
    df_aux = df.iloc[:, [0,1,2,3,4,6,7,8,9,11,12,13,14,15,16,17]]
    
    return df_aux
    
    
'''
Función encargada de llamar a los algoritmos de clustering
'''    
def realClustering1(datos):
    
    rAgglomerativeClustering1(datos)
    rKmeans1(datos)
    rKmedoids1(datos)
    #rMeanShift1(datos)
    rDbscan1(datos)
    rOptics1(datos)
    rSpectral1(datos)
    
    
    
    




def realClustering2(datos):
    
    #drawHeatMap(datos.iloc[:, :9])
    

    
    rAgglomerativeClustering2(datos)
    rKmeans2(datos)
    rKmedoids2(datos)
    rMeanShift2(datos)
    rDbscan2(datos)
    rOptics2(datos)
    rSpectral2(datos)
    
    