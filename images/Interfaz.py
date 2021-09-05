# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:25:52 2021

@author: sancappa
"""

import pandas as pd
from tkinter import *
from tkinter.ttk import *
import numpy as np
import matplotlib.pyplot as plt
from functools import partial  






        
'''
Funcion encargada de crear la interfaz gráfica y de recoger los datos
introducidos por el usuario
'''
def interfazCreation():
    #Creacción básica de la interfaz
    window = Tk()
    window.title("Configuración del espacio de puntos")
    window.geometry('450x400')
    
    
    #Creación de la zona de número de puntos
    labelNumPun = Label(window, text = "Introduce el número de puntos \n" +
                        "(Mínimo 50, Máximo 100)")
    labelNumPun.place(relx = 0.1, rely = 0.3)
    numPointsEntry = Entry(window, width = 10)
    numPointsEntry.place(relx = 0.1, rely = 0.5, width = 100 , height = 30)
    labelAux = Label(window, text = "Aux")
    
    
    #Creacción la zona de selección de epaciado entre puntos
    lblSelec = Label(window, text = "Selecciona el espaciado \n" + "entre puntos")
    lblSelec.place(relx=0.6, rely=0.3)
    combo = Combobox(window, state = 'readonly')
    combo['values']= ("Juntos", "Medio", "Separados", "Aleatorio")
    combo.current(3) #set the selected item
    combo.place(relx=0.6, rely=0.5, relwidth = 0.3, height=30)


    #Creaación boton acceptar
    #Funcion activada al clicar el boton
    def clicked():
        labelAux.configure(text = numPointsEntry.get())
        global aux1
        global aux2
        aux1 = numPointsEntry.get()
        aux2 = combo.get()
        window.destroy()
        
    btn = Button(window, text = "Aceptar", command = clicked)
    btn.place(relx = 0.4, rely = 0.8, width = 100, height = 50)

    
    #Activación de la interfaz
    window.mainloop()
    
    return (aux1, aux2)

        

'''
Funcion que se encarga de generar los puntos en el espacio
'''
def generatePoints(numPoints, distance):
    if(distance == "Juntos"):
        r_min = 0.3
        r_max = 3 + (100/numPoints)
        
    elif(distance == "Medio"):
        r_min = 80/(numPoints)
        r_max = (700/numPoints)
        
    elif(distance == "Separados"):
        r_min = 100 / numPoints 
        r_max = 1000
    else:
        r_min = 0
        r_max = 1000
        
    
    puntos = []
    
    #Introducimos 5 puntos aleatorios para facilitar la colocación del resto
    puntosBase = 1
    for i in range(0, puntosBase):
        x0,y0 = np.random.randint(0,100), np.random.randint(0,100)
        puntos.append([x0, y0])
     
    #Colocamos el resto de puntos
    while(i < numPoints):
        x,y = np.random.randint(0,100), np.random.randint(0,100)
        
        #Vemos que no haya ningun punto más cerca del radio mínimo ni mas lejos del maximo
        x_min_debajo = x - r_min
        x_min_encima = x + r_min
        y_min_izq = y - r_min
        y_min_der = y + r_min
        
        x_max_debajo = x - r_max
        x_max_encima = x + r_max
        y_max_izq = y - r_max
        y_max_der = y + r_max
        
        if ((minimumDistance(
            x_min_debajo, x_min_encima, y_min_izq, y_min_der, puntos)) and 
                (maximumDistance(
                    x_max_debajo, x_max_encima, y_max_izq, y_max_der, puntos))):
                    puntos.append([x, y])
                    i += 1
                
    return puntos
        
        
    
    
'''
Funcion auxiliar que sirve para comprobar que un nuevo punto está 
a una distancia mínima de cualquiera de los otros
'''
def minimumDistance(xa, xb, ya, yb, puntos):
    
    aux = []
    for p in puntos:
        if(xa < p[0] < xb and ya < p[1] < yb):
            aux.append(p)
   
    
    if(len(aux) > 0):
        return False
    else:
        return True


    
    
'''
Funcion auxiliar que sirve para comporbar que un nuevo punto no está
a más de una distancia máxima de todos los restantes
'''
def maximumDistance(xa, xb, ya, yb, puntos):
    aux = []
    for p in puntos:
        if(p[0] < xa or xb < p[0] or
            p[1] < ya or yb < p[1]):
            aux.append(p)
    
    if(len(aux) == len(puntos)):
        return False
    else:
        return True



'''
Funcion encargada de dibujar los puntos en el espacio
'''
def plotPoints(puntos, tipo):
    plt.title('Dibujo de ' + str(len(puntos)) + " puntos " + tipo)
    for p in puntos:
        plt.plot(p[0], p[1], 'r.')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()



    
    
    
