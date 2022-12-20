# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:15:52 2020

@author: pc
"""
import pandas as pd
import numpy as np
import time

#from OverlapManager import get_overlapRegions
from TableProcessing.OverlapManager import get_overlapRegions


    
#==============================================================================
# 
#==============================================================================
def add_overlapInformation(df_overlap, origin, overlap, imgDim):
        
    colNames = list(df_overlap.columns)
    colNames = colNames + ['overlapRef', 'overlapRegion', 'overlapID']              
    df_overlap = pd.DataFrame(df_overlap, columns=colNames)        

    xyzCells = df_overlap['X'].values, df_overlap['Y'].values, df_overlap['Z'].values
    v_mask, v_ref, v_region, v_regionID = get_overlapRegions(xyzCells, origin, overlap, imgDim)

    for i in range(0,len(v_mask)):
        df_overlap.loc[v_mask[i], ['overlapRef']] = get_xyzString(v_ref[i])
        df_overlap.loc[v_mask[i], ['overlapRegion']] = v_region[i]
        df_overlap.loc[v_mask[i], ['overlapID']] = v_regionID[i]  

    return df_overlap
    

def add_referenceCoordinates(df_Cells, origin):
    df_Cells['X_ref'] = origin[0]  
    df_Cells['Y_ref'] = origin[1] 
    df_Cells['Z_ref'] = origin[2]   
    return df_Cells
    
def add_absoluteCoordinates(df_Cells, origin, dissection_Ratio):
    print()
    print('add_absoluteCoordinates')
    print(dissection_Ratio)
    df_Cells['X_abs'] = (origin[0] + df_Cells['X'])*dissection_Ratio[0]
    df_Cells['Y_abs'] = (origin[1] + df_Cells['Y'])*dissection_Ratio[1]
    df_Cells['Z_abs'] = (origin[2] + df_Cells['Z'])*dissection_Ratio[2] 
    return df_Cells

    
#==============================================================================
# 
#==============================================================================
def get_xyzString(v):
    x, y, z = v
    xyz_str = 'x_' + str(int(x)) + '_y_' + str(int(y)) + '_z_' + str(int(z))

    return xyz_str

       

#==============================================================================
# Merged Non-Overlap Tables and Overlap Table
#==============================================================================
def merge_Tables(pathCenterDetections, pathOverlapDetections):
    
    n = len(pathCenterDetections)
    tablesCenter = []
    tablesOverlap = []
#    tablesTimes = []
    for i in range(0,n):
        df = pd.read_csv(pathCenterDetections[i], sep=';', header = 0, index_col = 0)
        tablesCenter.append(df)

        df = pd.read_csv(pathOverlapDetections[i], sep=';', header = 0, index_col = 0)
        tablesOverlap.append(df)
        
#        df = pd.read_csv(pathTimes[i], sep=';', header = 0, index_col = 0)
#        tablesTimes.append(df)
        
    mergedCenter  = pd.concat(tablesCenter, ignore_index=True)
    mergedOverlap = pd.concat(tablesOverlap, ignore_index=True)
#    mergedTimes = pd.concat(tablesTimes, ignore_index=True)
    
    return mergedCenter, mergedOverlap



#==============================================================================
# 
#==============================================================================
def unpacking_ParallelComputingResults(res):
    
    res = list(filter(None, res))
    M =  np.array(res, dtype=object)    
    M = np.concatenate(M, axis=0)
    
    df_Times    = pd.concat(M[:,0], axis=0, ignore_index=True)
    df_All      = pd.concat(M[:,1], axis=0, ignore_index=True)
    df_central  = pd.concat(M[:,2], axis=0, ignore_index=True)
    df_overlap  = pd.concat(M[:,3], axis=0, ignore_index=True)
    df_Origins  = pd.concat(M[:,4], axis=0, ignore_index=True)
    
    # M1, M2, M3, M4, M5 = M[:,0], M[:,1], M[:,2],  M[:,3], M[:,4]
      
    # #1) Unpacking: Computing Performance Data
    # M1 = np.concatenate(M1, axis=0)
    # df_Times = pd.DataFrame(M1, columns=['taskID', 'compID', 'processID', 'threadID', 'start', 'stop'])
    # df_Times['width'] = df_Times['stop'] - df_Times['start']

    
    # #2) Unpacking: Cell Detections in Central Regions
    # M2 = np.concatenate(M2, axis=0)

    # # col_names = ['X', 'Y', 'Z', 'S', 'I0', 'I1', 'I', 'dI', 'N',
    # #              'Vx', 'Vy', 'Vz', 'Ani', 'Tub', 'Disk',
    # #              'X_ref', 'Y_ref', 'Z_ref', 'X_abs', 'Y_abs', 'Z_abs']
    # col_names = M4[0].values
    # df_CentralDetections = pd.DataFrame(M2, columns=col_names)
    
    # #3) Unpacking: Cell Detections in Overlaping Region
    # M3 = np.concatenate(M3, axis=0)
    # # col_names = ['X', 'Y', 'Z', 'S', 'I0', 'I', 'I1', 'dI',  'N',
    # #              'Vx', 'Vy', 'Vz', 'Ani', 'Tub', 'Disk',
    # #              'X_ref', 'Y_ref', 'Z_ref', 'X_abs', 'Y_abs', 'Z_abs']
    # df_OverlapDetections = pd.DataFrame(M3, columns=col_names)       
    
    # return df_Times ,df_CentralDetections, df_OverlapDetections
    return df_Times, df_All, df_central, df_overlap, df_Origins

    
    
if __name__== '__main__':
    pass   





