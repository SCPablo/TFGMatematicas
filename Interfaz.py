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



def clicked():
    print("EEE")    
    #lbl.configure(text = "Configuración realizada. Calculando modelo")
        


def lonely(p,X,r):
    m = X.shape[1]
    x0,y0 = p
    x = y = np.arange(-r,r)
    x = x + x0
    y = y + y0

    u,v = np.meshgrid(x,y)

    u[u < 0] = 0
    u[u >= m] = m-1
    v[v < 0] = 0
    v[v >= m] = m-1

    return not np.any(X[u[:],v[:]] > 0)

        
def generate_samples(m=2500,r=200,k=30):
    # m = extent of sample domain
    # r = minimum distance between points
    # k = samples before rejection
    active_list = []

    # step 0 - initialize n-d background grid
    X = np.ones((m,m))*-1

    # step 1 - select initial sample
    x0,y0 = np.random.randint(0,m), np.random.randint(0,m)
    active_list.append((x0,y0))
    X[active_list[0]] = 1

    # step 2 - iterate over active list
    while active_list:
        i = np.random.randint(0,len(active_list))
        rad = np.random.rand(k)*r+r
        theta = np.random.rand(k)*2*np.pi

        # get a list of random candidates within [r,2r] from the active point
        candidates = np.round((rad*np.cos(theta)+active_list[i][0], rad*np.sin(theta)+active_list[i][1])).astype(np.int32).T

        # trim the list based on boundaries of the array
        candidates = [(x,y) for x,y in candidates if x >= 0 and y >= 0 and x < m and y < m]

        for p in candidates:
            if X[p] < 0 and lonely(p,X,r):
                X[p] = 1
                active_list.append(p)
                break
        else:
            del active_list[i]

    return X

X = generate_samples(2500, 200, 10)
s = np.where(X>0)
plt.plot(s[0],s[1],'.')

        

        

def main():
    

    window = Tk()
    window.title("Welcome to LikeGeeks app")
    
    window.geometry('450x400')
    
    
    #Creamos la zona de número de puntos
    labelNumPun = Label(window, text = "Introduce el número de puntos \n" +
                        "(Mínimo 20, Máximo 1000)")
    labelNumPun.place(relx = 0.1, rely = 0.3)
    
    txt = Entry(window, width = 10)
    txt.place(relx = 0.1, rely = 0.5, width = 100 , height = 30)

    
    btn = Button(window, text = "Aceptar", command = clicked)
    btn.place(relx = 0.4, rely = 0.8, width = 100, height = 50)
    
    
    #Creamos la zona de selección de epaciado entre puntos
    lblSelec = Label(window, text = "Selecciona el espaciado \n" + "entre puntos")
    lblSelec.place(relx=0.6, rely=0.3)
    combo = Combobox(window)
    combo['values']= ("Juntos", "Medio", "Separados", "Aleatorio")
    combo.current(3) #set the selected item
    combo.place(relx=0.6, rely=0.5, relwidth = 0.3, height=30)

    
    
    window.mainloop()
    
    
    
    generate_samples()
    



if __name__ == "__main__":
    main()
