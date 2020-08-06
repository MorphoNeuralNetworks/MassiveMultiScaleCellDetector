# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:15:52 2020

@author: pc
"""
import pandas as pd
import numpy as np

from OverlapManager import get_overlapRegions


    
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
    
def add_absoluteCoordinates(df_Cells, origin):
    df_Cells['X_abs'] = origin[0] + df_Cells['X'] 
    df_Cells['Y_abs'] = origin[1] + df_Cells['Y']
    df_Cells['Z_abs'] = origin[2] + df_Cells['Z']   
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
    for i in range(0,n):
        df = pd.read_csv(pathCenterDetections[i], sep=';', header = 0, index_col = 0)
        tablesCenter.append(df)

        df = pd.read_csv(pathOverlapDetections[i], sep=';', header = 0, index_col = 0)
        tablesOverlap.append(df)
        
    mergedCenter  = pd.concat(tablesCenter, ignore_index=True)
    mergedOverlap = pd.concat(tablesOverlap, ignore_index=True)
    
    return mergedCenter, mergedOverlap


#==============================================================================
#   Remove MultipleDetection in the Ovelap Table   
#==============================================================================

def remove_MultiDetectionInOverlapedRegions(df, v_xyz_iso, dissectionSize_iso, overlap_iso):   
    
    
    #Get the distance
    [dx, dy, dz] = dissectionSize_iso - overlap_iso
    
    #Get the Corner Cordinates of Dissected Volumes
    v_xyz_Corner = v_xyz_iso  - (dissectionSize_iso - 1)/2 

    #Loop 1: interate all volumes     
    n_volumes = v_xyz_Corner.shape[0]
    for i in range (0, n_volumes):  
        #Get all the Detected Cells from the Overlaping Regions of the Target Volume
        [x0, y0, z0] = v_xyz_Corner[i]
        mask = (df['X_ref']==x0)&(df['Y_ref']==y0)&(df['Z_ref']==z0)
        T_Local = df[mask]
        
        #Get all the Detected Cells from Overlaping Regions of all Neighbour Volumes
        vx = [x0-dx, x0, x0+dx]
        vy = [y0-dy, y0, y0+dy]
        vz = [z0-dz, z0, z0+dz]
        v = [vx, vy,vz]
        v_xyz_neighbour = np.array(np.meshgrid(*v)).T.reshape(-1, len(v)) 
        v_xyz_neighbour = np.delete(v_xyz_neighbour, (13), axis=0)
         
        T_neighbours = []
        for j in range(0, v_xyz_neighbour.shape[0]):        
            [x1, y1, z1] = v_xyz_neighbour[j]
            mask = (df['X_ref']==x1)&(df['Y_ref']==y1)&(df['Z_ref']==z1)
            T_neighbours.append(df[mask])       
    
        neighbours  = pd.concat(T_neighbours)

        #Loop 2: interate all Detected Cells
        n_cells = T_Local.shape[0]
        for k in range(0, n_cells):
            target = T_Local.iloc[k]
            R = target['S']
            [x0, y0, z0] = target['X_abs'], target['Y_abs'], target['Z_abs'] 
            [x1, y1, z1] = neighbours['X_abs'], neighbours['Y_abs'], neighbours['Z_abs']
            
            dr = np.sqrt((x1 - x0)**2 + (y1 -y0 )**2 + (z1 - z0)**2)
        
            #Condition    
            mask = dr<R 
            multiDetection = neighbours[mask] 
            multiDetection  =  multiDetection.append(target)
            
            [x0, y0, z0] = multiDetection['X'], multiDetection['Y'], multiDetection['Z'] 
            [x1, y1, z1] = (dissectionSize_iso - 1)/2 
            
            dr = np.sqrt((x1 - x0)**2 + (y1 -y0 )**2 + (z1 - z0)**2)
            mask = dr!=dr.min()
            
            # Check if there are more than one            
            if (mask==False).sum()>1:
                ix = np.where(dr==dr.min())[0]
                mask.iloc[ix[1:]]=True
           
#            # Check if there are more than one           
#            ix = np.where(dr==dr.min())[0]
#            if ix.shape[0]>1:                      
#                mask.iloc[ix[1:]]=True
                
           
            #remove the Multiple Detection
            ix = mask.index
            ix = ix[mask]
            df = df.drop(ix)
            
    return df
            
            



def remove_MultiDetectionInOverlapedRegions2(mergedOverlapTable):
    
    df = mergedOverlapTable.copy()
    
    #Loop for all overlapRef
    v_overlapRef= df['overlapRef'].unique()
    nRef = v_overlapRef.shape[0]
    for i in range(0, nRef):
        #Get the Reference of an Overlap Region
        mask = (df['overlapRef']==v_overlapRef[i])
        df_Ref = df[mask]
        
        #Loop for all overlapRegions of the same overlapRef (corner, side, bar etc)
        v_overlapRegion = df_Ref['overlapRegion'].unique()
        nRegion = v_overlapRegion.shape[0]
        for j in range(0,nRegion):  
            #Get the Type of the Overlaping Region
            mask = (df_Ref['overlapRegion']==v_overlapRegion[j])
            df_Region = df_Ref[mask]

            #Mask more than 1 detection in the same overlaping region
            #pending to integrate
            #For example:
            #if corner then 0<n<=8 -> 1 detection
            #if corner then 8<n<16 -> 2 detection           
            
            #Remove the Multiple Detection                      
            mask = (df_Region['I']==df_Region['I'].max())
            ixSave = df_Region[mask].index[0]
            mask = (ixSave!=df_Region.index)
            ixRemove = df_Region[mask].index
            df = df.drop(ixRemove)  
    
    return df



    
    
if __name__== '__main__':
    pass   





