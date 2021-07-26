# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:25:52 2021

@author: sancappa
"""

import pandas as pd
from tkinter import *
from tkinter.ttk import *



def clicked():
        lbl.configure(text = "Configuración realizada. Calculando modelo")

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
        
    



if __name__ == "__main__":
    main()
