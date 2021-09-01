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
def experiment():
    information = interfazCreation()
    
    while((not (information[0].isdigit())) or (int(information[0]) < 50) or 
          (int(information[0]) > 100)):
        print("Introduce un número entre 45 y 100")
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
    datasets = ["Songs", "Corona"]
    dataset = datasets[1]
    
    if(dataset == "Songs"):
        df = readFile("Datasets/Canciones.csv")
        df = preprocessSongs(df)
        realClustering1(df)
        
    elif(dataset == "Corona"):
        df = readFile("Datasets/Corona.csv")
        df = preprocessCorona(df)
        realClustering2(df.iloc[:10000, :])
        
    
    
    
'''
Funcion principal del proyecto
'''
def main():
    modes = ["Real Datasets", "Experiment"]
    mode = modes[1]

    

    if(mode == "Real Datasets"):
        realDataset()
    elif(mode == "Experiment"):
        experiment()
    



if __name__ == "__main__":
    main()
