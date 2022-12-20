# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 19:20:03 2020

@author: pc
"""

import time
import pandas as pd
import numpy as np

#==============================================================================
# 
#==============================================================================
def get_centralMask(xyzCells, origin, overlap, imgDim):
    #Unpacking
    x, y, z = xyzCells
    x0, y0, z0 = overlap
    x1, y1, z1 = imgDim - overlap
    
    #Region: Non-Overlaping Region (i.e. Central Volume)
    boolMask = ((x >= x0)&(x <= x1)&
                (y >= y0)&(y <= y1)&
                (z >= z0)&(z <= z1)
                )
    return boolMask

#==============================================================================
#     
#==============================================================================
def get_overlapMask(xyzCells, origin, overlap, imgDim):
    #Unpacking
    x, y, z = xyzCells
    x0, y0, z0 = overlap
    x1, y1, z1 = imgDim - overlap
    
    #Region: Overlaping Regions
    boolMask = ((x < x0)|(x > x1)|
                (y < y0)|(y > y1)|
                (z < z0)|(z > z1)
                ) 
    return boolMask


#==============================================================================
#   Remove MultipleDetection in the Ovelap Table   
#==============================================================================
def remove_MultipleDetection(df_Overlap, df_Origins, scannerSize_Out_px):
    start = time.time()
    # Save a BackUp 
    df_Overlap_In = df_Overlap.copy()    

    # Create an empty DataFrame to fill in without MultipleDetections
    df_Out = pd.DataFrame()
    
    # Get all Origins (aka refs)
    df_Cubes = df_Origins.groupby(['X_ref_out_px', 'Y_ref_out_px', 'Z_ref_out_px']).size().reset_index().rename(columns={0:'count'})
    # df_Cubes = df_Overlap_In.groupby(['X_ref_out_px', 'Y_ref_out_px', 'Z_ref_out_px']).size().reset_index().rename(columns={0:'count'})
    
    # ??? Loop: Select one Overlaping Volume belonging to a Cube
    n_Cubes = df_Cubes.shape[0]
    for i in range(0, n_Cubes): # i = 0
        
        #Select a Current Cube
        df_Cube = df_Cubes.iloc[i]
        
        #Get the Cubes Around (i.e. the closest neighbours of the Current Cube)
        r = np.sqrt((df_Cubes['X_ref_out_px'] - df_Cube['X_ref_out_px'])**2 + 
                    (df_Cubes['Y_ref_out_px'] - df_Cube['Y_ref_out_px'])**2 +
                    (df_Cubes['Z_ref_out_px'] - df_Cube['Z_ref_out_px'])**2 
                    )
        #Note:
        # the minimum distance to the next cube is the side of the cube (l)
        # but the maximum distance to the next cube is the diagonal (h=sqrt(l**2 + l**2)=sqrt(2)*l)
        R = scannerSize_Out_px.mean()
        R = 1.2*(np.sqrt(2)*R)
        mask = (r>0) & (r<R)
        df_CubesAround = df_Cubes[mask]

        #Select Detections from the Cube 
        mask = ((df_Overlap_In['X_ref_out_px'] == df_Cube['X_ref_out_px']) &
                (df_Overlap_In['Y_ref_out_px'] == df_Cube['Y_ref_out_px']) &
                (df_Overlap_In['Z_ref_out_px'] == df_Cube['Z_ref_out_px']) )    
        df_CubeDetections = df_Overlap_In[mask]
        
        # ??? loop
        df_CubeAroundDetections = pd.DataFrame()
        n_CubesAround = df_CubesAround.shape[0]
        if n_CubesAround!=0:
            for j in range(0, n_CubesAround): # j = 0
                
                # Select one Cube Around
                df_CubeAround = df_CubesAround.iloc[j]
                
                # Select Detections from the (Current) Around Cube
                mask = ((df_Overlap_In['X_ref_out_px'] == df_CubeAround['X_ref_out_px']) &
                        (df_Overlap_In['Y_ref_out_px'] == df_CubeAround['Y_ref_out_px']) &
                        (df_Overlap_In['Z_ref_out_px'] == df_CubeAround['Z_ref_out_px']) ) 
                df_CubeAroundDetections = df_CubeAroundDetections.append(df_Overlap_In[mask])
      
                
            # ????? Loop
            ix = []
            ix = pd.DataFrame()
    
            n_CubeDetections = df_CubeDetections.shape[0]
            for k in range(0, n_CubeDetections): # k = 0
                
                # Select One Detection    
                df_CubeDetection = df_CubeDetections.iloc[k]
            
                #Select a Detection with a inner detection
                r = np.sqrt((df_CubeAroundDetections['X_abs_out_px'] - df_CubeDetection['X_abs_out_px'])**2 + 
                            (df_CubeAroundDetections['Y_abs_out_px'] - df_CubeDetection['Y_abs_out_px'])**2 +
                            (df_CubeAroundDetections['Z_abs_out_px'] - df_CubeDetection['Z_abs_out_px'])**2 
                            )
                mask = (r<=df_CubeDetection['S']) 
                df_Repited = df_CubeAroundDetections[mask]
                
                # Merge   
                df_Repited = df_Repited.append(df_CubeDetection, ignore_index=False)
                
                # Get only those MultipleDetection with the highest I_DoG and stored at df_out
                mask = (df_Repited['I_DoG']==df_Repited['I_DoG'].max())
                df_Unique = df_Repited[mask]
                df_Out = df_Out.append(df_Unique)
                
                # Remove Processed Points from df_Overlap to avoid process them multiple times (to speed up the loops)
                # df_Overlap_In = df_Overlap_In.drop(df_Repited.index)
                ix = ix.append(list(df_Repited.index))
                
            # jajaj
            ix = np.unique(ix)
            df_Overlap_In = df_Overlap_In.drop(list(ix))
                        
                # Verbose
                # print()
                # print('k=',k)
                # print(r.min(), r.max())
                # print(df_Repited[['X_abs_out_px', 'Y_abs_out_px', 'Z_abs_out_px', 'S', 'I_DoG']])
                # print(df_Repited[['X_ref_out_px', 'Y_ref_out_px', 'Z_ref_out_px']])
                
    print()
    print('End')
    
    print()
    print('Check: The DataFrame should be Empty')
    print(df_Overlap_In)
    # df_OverlapOut = df_Out.drop_duplicates(subset=['X_abs_out_px', 'Y_abs_out_px', 'Z_abs_out_px'])
    
    #Time
    stop = time.time()
    dt_removeMultiDetections = stop - start
    
    print()
    print('dt_removeMultiDetections \n', dt_removeMultiDetections)

    return df_Out


#==============================================================================
# 
#==============================================================================
#        #Test: yBar
#        df_overlap.loc[6,'X'] = 11
#        df_overlap.loc[6,'Y'] = 20
#        df_overlap.loc[6,'Z'] = 11
#        
#        #Test xSide
#        df_overlap.loc[6,'X'] = 11
#        df_overlap.loc[6,'Y'] = 20
#        df_overlap.loc[6,'Z'] = 20 
#
#        add_overlapInformation(df_overlap, origin, overlap, imgDim)

def get_overlapRegions(xyzCells, origin, overlap, imgDim):
    #Unpacking
    x, y, z = xyzCells
    x0, y0, z0 = overlap
    x1, y1, z1 = imgDim - overlap
    
    #Reference System: 8 points of the cube (in absolute Coordinates)        
    v000 = origin + [0*x1, 0*y1, 0*z1 ]
    v100 = origin + [1*x1, 0*y1, 0*z1 ]
    v010 = origin + [0*x1, 1*y1, 0*z1 ]
    v110 = origin + [1*x1, 1*y1, 0*z1 ]
    
    v001 = origin + [0*x1, 0*y1, 1*z1 ]
    v101 = origin + [1*x1, 0*y1, 1*z1 ]
    v011 = origin + [0*x1, 1*y1, 1*z1 ]
    v111 = origin + [1*x1, 1*y1, 1*z1 ]
  
    #Initializations
    v_mask = []
    v_ref  = []
    v_region = []
    v_regionID   = []
    
    
    #Region: Corners (x8)           
    boolMask000 = ( (x < x0)&
                    (y < y0)&
                    (z < z0)
                    )
    boolMask100 = ( (x > x1)&
                    (y < y0)&
                    (z < z0)
                    )
    boolMask010 = ( (x < x0)&
                    (y > y1)&
                    (z < z0)
                    )          
    boolMask110 = ( (x > x1)&
                    (y > y1)&
                    (z < z0)
                    )     
    boolMask001 = ( (x < x0)&
                    (y < y0)&
                    (z > z1)
                    )
    boolMask101 = ( (x > x1)&
                    (y < y0)&
                    (z > z1)
                    )
    boolMask011 = ( (x < x0)&
                    (y > y1)&
                    (z > z1)
                    )
    boolMask111 = ( (x > x1)&
                    (y > y1)&
                    (z > z1)
                    )
    
    mask = [boolMask000, boolMask100, boolMask010, boolMask110,
            boolMask001, boolMask101, boolMask011, boolMask111]
    ref = [v000, v100, v010, v110,
           v001, v101, v011, v111]
    regionID  = ['000', '100', '010', '110',
                  '001', '101', '011', '111']    
    region = len(mask)*['corner']
    
    v_mask = v_mask + mask
    v_ref  = v_ref  + ref
    v_regionID   = v_regionID   + regionID
    v_region = v_region + region
    
    
    #Region: Ybars           
    boolMask000 = ( (x < x0)&
                    (y >= y0)&(y <= y1)&
                    (z < z0)
                    )
    boolMask100 = ( (x > x1)&
                    (y >= y0)&(y <= y1)&
                    (z < z0)
                    )            
    boolMask001 = ( (x < x0)&
                    (y >= y0)&(y <= y1)&
                    (z > z1)
                    )
    boolMask101 = ( (x > x1)&
                    (y >= y0)&(y <= y1)&
                    (z > z1)
                    )
    
    mask = [boolMask000, boolMask100,
                   boolMask001, boolMask101]
    ref = [v000, v100,
                  v001, v101]
    regionID  = ['000', '100',
                  '001', '101']    
    region = len(mask)*['Ybar']
    
    v_mask = v_mask + mask
    v_ref  = v_ref  + ref
    v_regionID   = v_regionID   + regionID
    v_region = v_region + region
                  
    #Region: Xbars           
    boolMask000 = ( (x >= x0)&(x <= x1)&
                    (y < y0)&
                    (z < z0)
                    )
    boolMask010 = ( (x >= x0)&(x <= x1)&
                    (y > y1)&
                    (z < z0)
                    )              
    boolMask001 = ( (x >= x0)&(x <= x1)&
                    (y < y0)&
                    (z > z1)
                    )
    boolMask011 = ( (x >= x0)&(x <= x1)&
                    (y > y1)&
                    (z > z1)
                    )
    
    mask = [boolMask000, boolMask010,
                   boolMask001, boolMask011]
    ref = [v000, v010, 
                  v001, v011]
    regionID  = ['000', '010',
                  '001', '011']    
    region = len(mask)*['Xbar']
    
    v_mask = v_mask + mask
    v_ref  = v_ref  + ref
    v_regionID   = v_regionID   + regionID
    v_region = v_region + region
 
    #Region: Zbars           
    boolMask000 = ( (x < x0)&
                    (y < y0)&
                    (z >= z0)&(z <= z1)
                    )
    boolMask100 = ( (x > x1)&
                    (y < y0)&
                    (z >= z0)&(z <= z1)
                    )
    boolMask010 = ( (x < x0)&
                    (y > y1)&
                    (z >= z0)&(z <= z1)
                    )          
    boolMask110 = ( (x > x1)&
                    (y > y1)&
                    (z >= z0)&(z <= z1)
                    )                  
    
    mask = [boolMask000, boolMask100, boolMask010, boolMask110]
    ref = [v000, v100, v010, v110]
    regionID  = ['000', '100', '010', '110']    
    region = len(mask)*['Zbar']
    
    v_mask = v_mask + mask
    v_ref  = v_ref  + ref
    v_regionID   = v_regionID   + regionID
    v_region = v_region + region
    
    #Region: Xside          
    boolMask000 = ( (x < x0)&
                    (y >= y0)&(y <= y1)&
                    (z >= z0)&(z <= z1)
                    )
    boolMask100 = ( (x > x1)&
                    (y >= y0)&(y <= y1)&
                    (z >= z0)&(z <= z1)
                    )
                           
    mask = [boolMask000, boolMask100]
    ref = [v000, v100]
    regionID  = ['000', '100']    
    region = len(mask)*['Xside']
    
    v_mask = v_mask + mask
    v_ref  = v_ref  + ref
    v_regionID   = v_regionID   + regionID
    v_region = v_region + region
                        
    #Region: Yside
    boolMask000 = ( (x >= x0)&(x <= x1)&
                    (y < y0)&
                    (z >= z0)&(z <= z1)
                    )                       
    boolMask010 = ( (x >= x0)&(x <= x1)&
                    (y > y1)&
                    (z >= z0)&(z <= z1)
                    )                        

    
    mask = [boolMask000, boolMask010]
    ref = [v000, v010]
    regionID  = ['000','010']    
    region = len(mask)*['Yside']
    
    v_mask = v_mask + mask
    v_ref  = v_ref  + ref
    v_regionID   = v_regionID   + regionID
    v_region = v_region + region
                        
    #Region: Zside
    boolMask000 = ( (x >= x0)&(x <= x1)&
                    (y >= y0)&(y <= y1)&
                    (z < z0)
                    ) 
    boolMask001 = ( (x >= x0)&(x <= x1)&
                    (y >= y0)&(y <= y1)&
                    (z > z1)
                    ) 

    
    mask = [boolMask000, boolMask001]
    ref = [v000, v001]
    regionID  = ['000', '001']    
    region = len(mask)*['Zside']
    
    v_mask = v_mask + mask
    v_ref  = v_ref  + ref
    v_regionID   = v_regionID   + regionID
    v_region = v_region + region


    return v_mask, v_ref, v_region, v_regionID 
                       
if __name__== '__main__':
    pass        