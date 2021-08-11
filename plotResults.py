#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:02:13 2021

@author: pablo
"""


from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import umap
import plotly.express as px


'''
Función encargada de dibujar los puntos
'''

def plotResultsGrafics(puntos, clustering, name):
    plt.title(name)


    scatter = plt.scatter(puntos[:,0], puntos[:,1], c=clustering.labels_,
                cmap='rainbow')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    legend_list = []
    for i in range(0, len(np.unique(clustering.labels_))):
        legend_list.append(str(i))
    
    if(len(legend_list) < 10):
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=legend_list,  fancybox=True, shadow=True, 
                   loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=9)
    
    

    
    plt.show()



def plotResultTable(datos, clusters, algorithm):
    aux = pd.DataFrame()
    aux['Cluster']=clusters.labels_

    aux['Genre'] = datos['Genre']
    
    
    
    #Generate colors randomly
    colors = ['lightgoldenrodyellow','lightgray', 'lightgreen', 
              'lightpink', 'lightsalmon', 'lightseagreen',
                'lightskyblue', 'lightyellow', 'lime', 'linen',
                'magenta',  'mediumaquamarine']

    colors_final= []
    for c in clusters.labels_:
        colors_final.append(colors[int(c)%len(colors)])

    aux['Color'] = colors_final
   
    mostrar = pd.DataFrame()
    mostrar['result'] = aux.groupby(['Cluster', 'Color','Genre']).size()
    
    fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    specs=[[{"type": "table"}],
           [{"type": "table"}]]
    )
    
    fig.add_trace(
    go.Table(
        columnwidth = [60,30],
        header=dict(
            values=[['<b>Genre</b>'], ['<b>Número</b>']]
        ),
        cells=dict(
            values=[mostrar.reset_index()['Genre'],
                    mostrar.reset_index()['result']],
            fill_color=[mostrar.reset_index()['Color']],
            align = "left")
    ),
    row=1, col=1
    )
    
    fig.update_layout(width=500, height=1300, title =  algorithm)
    fig.write_html("images/" + "Clustering " + algorithm + ".html")
    fig.show()
    
    
    
    
    
def drawHeatMap(datos):
    sns.heatmap(datos.corr(), linewidths=.2,
                cmap="YlOrRd")
    
    



def reduceDimension(datos, clusters, titulo):
    reducer = umap.UMAP(random_state=42)
    
    numeric_numpy = datos.drop(columns = ['cluster']).to_numpy()
    embedding = reducer.fit_transform(numeric_numpy)
    
    
    colors = ['lightgoldenrodyellow','lightgray', 'lightgreen', 
              'lightpink', 'lightsalmon', 'lightseagreen',
                'lightskyblue', 'lightyellow', 'lime', 'linen',
                'magenta',  'mediumaquamarine']

    colors_final= []
    for c in clusters:
        colors_final.append(colors[int(c)%len(colors)])
    
    x = pd.DataFrame(data={'x_axis':embedding[:, 0]})
    y = pd.DataFrame(data={'y_axis':embedding[:, 1]})
    
    plt.scatter(x, y, c=datos.cluster, cmap='YlOrRd')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar().set_ticks(np.unique(clusters))
    plt.title('UMAP ' + titulo, fontsize=14);
    plt.show()
    




    