# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 18:07:24 2020

@author: pc
"""



import os
import pandas as pd
import numpy as np

def save_CSV(df, folderPath, fileName):
    
    #Create Folder
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)  
    
    #Saving the Table
    fileExtension = '.csv'
    filePath = os.path.join(folderPath, fileName + fileExtension)  
    df.to_csv(filePath, sep=';', encoding='utf-8', index=True)  
    

def save_Vaa3DMarker(df_Cells, imgDim, folderPath, fileName):

    #Create Folder
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)    
        
    #Creater the Vaa3D Marker Format
    df = pd.DataFrame()
    n = df_Cells.shape[0]
    
    #Change the Axis Reference System
    nx, ny, nz = imgDim
    df['X'] = df_Cells['Z']  
    df['Y'] = nx - df_Cells['X'] 
    df['Z'] =  df_Cells['Y'] 
    
    
    
    df['R'] = 100*df_Cells['S']     
    df['shape'] = np.zeros(n)
    df['name'] =  df_Cells.index.values
    df['comment'] = n*['0']
    df['cR'] = 255*np.ones(n)
    df['cG'] = 0*np.ones(n)
    df['cB'] = 0*np.ones(n)
    
    df = df.astype(int)
    
    #Saving the Marker File
    fileExtension = '.marker'
    filePath = os.path.join(folderPath, fileName + fileExtension)  
    df.to_csv(filePath, sep=',', encoding='utf-8', index=False, header = False)    

def save_Figure(fig, folderPath, fileName):
    #Saving the Matplotlib Figure
    graph_dpi = 150
    fileExtension = '.png'
    filePath = os.path.join(folderPath, fileName + fileExtension)
    fig.savefig(filePath, dpi=graph_dpi, bbox_inches='tight')
    
   
        
        
        
        
if __name__== '__main__':
    pass