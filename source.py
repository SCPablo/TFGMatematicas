#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 12:16:04 2021

@author: pablo
"""

from Interfaz import interfazCreation, generatePoints, plotPoints
from clustering import experimentClustering, realClustering1, realClustering2, preprocessSongs, preprocessCorona
import numpy as np
import pandas as pd
import csv




'''
Funcion encargada de elaborar el experimento. Crea la interfaz,
luego el conjunto de puntos y luego los agrupa
'''
def experiments():
    information = interfazCreation()
    
    while((not (information[0].isdigit())) or (int(information[0]) < 50) or 
          (int(information[0]) > 100)):
        print("Introduce un número entre 40 y 100")
        information = interfazCreation()
    
    
    
    puntos = generatePoints(int(information[0])-1, information[1])
    plotPoints(puntos, information[1])
    
    experimentClustering(np.array(puntos), information[1])




'''
Funcion encargada de leer el archivo de datos
'''
def readFile(nombre):
    df = pd.read_csv(nombre, sep =';')
 
    return df
    


'''
Funcion encargada de leer el archivo de datos y realizar el agrupamiento
sobre sus datos
'''
def realDataset():
    datasets = ["Songs", "Corona", "Both"]
    dataset = datasets[1]
    
    if(dataset == "Songs"):
        df = readFile("Datasets/Canciones.csv")
        df = preprocessSongs(df)
        realClustering1(df)
        
    elif(dataset == "Corona"):
        df = readFile("Datasets/Corona.csv")
        df = preprocessCorona(df)
        realClustering2(df.iloc[:10000, :])
        
    else:
        df1 = readFile("Datasets/Canciones.csv")
        df1 = preprocessSongs(df1)
        realClustering1(df1)
        df2 = readFile("Datasets/Corona.csv")
        df2 = preprocessCorona(df2)
        realClustering2(df2)
    
    
    

def main():
    modes = ["Real Datasets", "Experiments", "Both"]
    mode = modes[0]

    

    if(mode == "Real Datasets"):
        realDataset()
    elif(mode == "Experiments"):
        experiments()
    else:
        realDataset()
        experiments()



if __name__ == "__main__":
    main()
