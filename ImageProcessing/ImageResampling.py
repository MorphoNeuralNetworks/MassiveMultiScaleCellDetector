# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 18:25:13 2020

@author: pc
"""

from scipy import signal
import numpy as np
#==============================================================================
# 
#==============================================================================
def resample_Zdim(imgIn, z_thick, dissectionSize_iso, overlap_iso):
    
    #Forcing a 3D-Image to be Isotropic through a Resampling Operator    
    if z_thick>1.0:   
        zDim = imgIn.shape[2]*z_thick        
        Nz = (zDim-overlap_iso[0])/(dissectionSize_iso[0]-overlap_iso[0])
        Nz = int(np.round(Nz))
        zDim = Nz*(dissectionSize_iso[0]-overlap_iso[0]) + overlap_iso[0]
    
        imgIn = signal.resample(imgIn, zDim, axis=2)
        
    return imgIn


        
if __name__== '__main__':
    pass