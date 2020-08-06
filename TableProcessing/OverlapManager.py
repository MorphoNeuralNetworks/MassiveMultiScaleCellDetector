# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 19:20:03 2020

@author: pc
"""


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