# -*- coding: utf-8 -*-
"""
Created on Sat May 09 15:41:47 2020

@author: pc
"""
import sys
import os
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm

def get_ImagePatch(img3D, x, y, z, dx, dy, dz):
    
    #Get Dimensions of Original 3D image
    [ny, nx, nz] = img3D.shape    

    #Get Extreme Index to extract the Image Patch from the Whole Image
    x0, x1 = get_CenteredExtremes(x, dx)
    y0, y1 = get_CenteredExtremes(y, dy)
    z0, z1 = get_CenteredExtremes(z, dz)    
     
    #Get 3D image Patch from the Big 3D Image Sequence
    img3DPatch = img3D[y0:y1+1, x0:x1+1, z0:z1+1]
    
    return img3DPatch  
    
    

 
def get_CenteredExtremes(r, dr):
    dr_half = dr//2    
    [r0, r1 ] = [r - dr_half, r + dr_half]
    if r0<0:
        r0 = 0
    return r0, r1

def get_Odd(num):
    if (num % 2) == 0: 
        num = num + 1
    return num


if __name__== '__main__':
       
    
    #Set the filePath to Read 3D numpy Array    
    locaPath = 'D:\\MyPythonPosDoc\\P4_Cell_Detection'
    locaPath = os.path.dirname(sys.argv[0])
    locaPath = Path(locaPath)
    locaPath = locaPath.parent
    locaPath = str(locaPath.absolute())
    saveFolder = 'TestData'
    fileName = 'img3D_0.npy'
    readPath = os.path.join(locaPath, saveFolder, fileName) 
    
    #Get Image from the path
    imgIn = np.load(readPath) 
#==============================================================================
#     
#==============================================================================
    x, y, z = 200, 200, 200    
    x, y, z = 50, 50, 50 
#    x, y, z = 10, 10, 10
#    x, y, z = 5, 5, 5
#    x, y, z = 95, 95, 95
    n = 21
    dx, dy, dz = n, n, n 
#==============================================================================
#     
#==============================================================================
    imgPatch = get_ImagePatch(imgIn, x, y, z, dx, dy, dz)
    
#==============================================================================
#     
#==============================================================================
    [ny, nx, nz] = imgIn.shape    
    plt.imshow(imgIn[:, :,nz//2], cm.Greys_r, interpolation='nearest')
    plt.show()
    
    [ny, nx, nz] = imgPatch.shape
    print(imgPatch.shape)
    plt.imshow(imgPatch[:, :,nz//2], cm.Greys_r, interpolation='nearest')
    plt.show()
    
    
    
    
    
    
    
    
