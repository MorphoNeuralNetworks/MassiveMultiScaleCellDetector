# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 18:07:24 2020

@author: pc
"""



import os
import sys

import pandas as pd
import numpy as np


def save_CSV(df, folderPath, fileName, sep=',', index=True, encoding='utf-8'):
    
    #Create Folder
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)  
    
    #Saving the Table
    fileExtension = '.csv'
    filePath = os.path.join(folderPath, fileName + fileExtension) 
    df.to_csv(filePath, sep=sep, encoding='UTF-8', index=index)
    #df.to_csv(filePath, sep=';', encoding='utf-8', index=True)  
    # df.to_csv(filePath, sep=',', encoding='utf-8', index=True) 
    

def save_Vaa3DMarker(df_Cells, imgDim, folderPath, fileName):

    #Create Folder
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)    
        
    #Creater the Vaa3D Marker Format
    df = pd.DataFrame()
    n = df_Cells.shape[0]
    
    #Change the Axis Reference System
    nx, ny, nz = imgDim
    
    #Case 1: Python version 2
    if sys.version_info[0]==2:
        df['X'] = df_Cells['Z'] + 1
#        df['Y'] = (nx - df_Cells['X']) + 1
        df['Y'] = df_Cells['X'] + 1
        df['Z'] = df_Cells['Y'] + 1
    #Case 2: Python version 3
    elif  sys.version_info[0]==3: 
        df['X'] =  -0*nz + (df_Cells['Z'] )
        df['Y'] =  -0*nx + (df_Cells['X'] )  
        df['Z'] =  -0*ny + (df_Cells['Y'] )    
    else:
        print('Unknow Python Version') 
 
    # df['X'] =  -0*nz + (df_Cells['Z_abs']/dissection_Ratio[2] - origin[2])
    # df['Y'] =  -0*nx + (df_Cells['X_abs']/dissection_Ratio[0] - origin[0])
    # df['Z'] =  -0*ny + (df_Cells['Y_abs']/dissection_Ratio[1] - origin[1])  
    
    df['R'] = 100*df_Cells['S']     
    df['shape'] = np.zeros(n)
    df['name'] =  df_Cells.index.values
    df['comment'] = n*['0']
    df['cR'] = 255*np.ones(n)
    df['cG'] = 0*np.ones(n)
    df['cB'] = 0*np.ones(n)
    
    df = df.astype(int)

    print('')
    print(df_Cells[['Z', 'X', 'Y', 'Z_abs', 'X_abs', 'Y_abs']])
    print(df[['X', 'Y', 'Z']])
    
    #Saving the Marker File
    fileExtension = '.marker'
    filePath = os.path.join(folderPath, fileName + fileExtension)  
    df.to_csv(filePath, sep=',', encoding='utf-8', index=False, header = False)    


def save_Vaa3DMarker_Abs(df_Cells, origin, imgDim, folderPath, fileName, dissection_Ratio):

    #Create Folder
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)    
        
    #Creater the Vaa3D Marker Format
    df = pd.DataFrame()
    n = df_Cells.shape[0]
    
    #Change the Axis Reference System
    nx, ny, nz = imgDim   
    
    
    #Change the Axis Reference System
    df['X'] =  -0*nz + (df_Cells['Z_abs']/dissection_Ratio[2] - origin[2])
    df['Y'] =  -0*nx + (df_Cells['X_abs']/dissection_Ratio[0] - origin[0])
    df['Z'] =  -0*ny + (df_Cells['Y_abs']/dissection_Ratio[1] - origin[1])
        
    #Other Features
    df['R'] = 100*df_Cells['S']     
    df['shape'] = np.zeros(n)
    df['name'] =  df_Cells.index.values
    df['comment'] = n*['0']
    df['cR'] = 255*np.ones(n)
    df['cG'] = 0*np.ones(n)
    df['cB'] = 0*np.ones(n)
    
    df = df.astype(int)
    
#    print('')
#    print(df_Cells[['Z', 'X', 'Y', 'Z_ref', 'X_ref', 'Y_ref', 'Z_abs', 'X_abs', 'Y_abs']])
#    print(df[['X', 'Y', 'Z']])

    print('')
    print(df_Cells[['X', 'Y', 'Z', 'X_ref', 'Y_ref', 'Z_ref', 'X_abs', 'Y_abs', 'Z_abs']])
    print(df[['X', 'Y', 'Z']])    
    
    #Saving the Marker File
    fileExtension = '.marker'
    filePath = os.path.join(folderPath, fileName + fileExtension)  
    df.to_csv(filePath, sep=',', encoding='utf-8', index=False, header = False) 
    
    df['X_abs'] = df_Cells['X_abs']
    df['Y_abs'] = df_Cells['Y_abs']
    df['Z_abs'] = df_Cells['Z_abs']
    df['Vx'] = df_Cells['Vx']
    df['Vy'] = df_Cells['Vy']
    df['Vz'] = df_Cells['Vz']
    df['I0'] = df_Cells['I0']
    df['I'] = df_Cells['I']
    df['dI'] = df_Cells['dI']
    df['Tub'] = df_Cells['Tub']
    df['S'] = df_Cells['S']
    df['R_um'] = df_Cells['R_um']
    df['N'] = df_Cells['N']
    
    fileExtension = '.csv'
    filePath = os.path.join(folderPath, fileName + fileExtension)  
    #df.to_csv(filePath, sep=';', encoding='utf-8', index=True)  
    df.to_csv(filePath, sep=',', encoding='utf-8', index=True) 
    
    
def save_Figure(fig, folderPath, fileName):
    #Saving the Matplotlib Figure
    graph_dpi = 150
    fileExtension = '.png'
    filePath = os.path.join(folderPath, fileName + fileExtension)
    fig.savefig(filePath, dpi=graph_dpi, bbox_inches='tight')
    
   
        
        
        
        
if __name__== '__main__':
    pass