# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:12:15 2020

@author: pc
"""

#Maths Operation
import numpy as np

import os
import sys

# import cv2

#Ploting Library
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def change_ImageIntensityMap(A, x0, x1, y0, y1):  
    #Masking Saturation 
    A[A<x0] = x0   
    A[A>x1] = x1 
    
    #Mapping Conversion
    B = (A-x0)*((y1-y0)/float(x1 - x0)) + y0
    return B 
#    B = (((A-lim_inf)/float(lim_sup-lim_inf))*(new_sup-new_inf)+new_inf)
   
#     B = np.round()
#    B = np.uint8(B)       
if __name__== '__main__':  
    
    img = np.asarray([265, 500])
    bitDepth = 16
    change_ImageIntensityMap(img, x0=0, x1=2**bitDepth-1, y0=-1, y1=+1)
    jaja
    
    #Set the filePath to save    
    locaPath = 'D:\\MyPythonPosDoc\\P4_Cell_Detection'
    imgFolder = 'TestData'
    imgName = 'img2D_0.tif'
    imgPath = os.path.join(locaPath, imgFolder, imgName)    
    # img = cv2.imread(imgPath, -1)
    
#==============================================================================
#     
#==============================================================================
    img2 = change_ImageIntensityMap(img, x0=0, x1=255, y0=-1, y1=+1)
    
#==============================================================================
#     
#==============================================================================
    plt.imshow(img,  cm.Greys_r, interpolation='nearest') 
    plt.show()
    plt.imshow(img2,  cm.Greys_r, interpolation='nearest') 
    plt.show()
#==============================================================================
#     
#==============================================================================
    print('Min=', img.min())
    print('Max=', img.max())
    print('Min=', img2.min())
    print('Max=', img2.max())


#==============================================================================
#   Draft
#==============================================================================
    
    #    
##From [0...128...255] uint8 to [-1..0..1]  float64  
#def Convert255ToBalanced(A): 
#    #Masking Saturation    
#    A[A>255] = 255
#    A[A<0] = 0
#    
#    lim_sup = 255
#    lim_inf = 0
#    
#    new_sup = 1
#    new_inf = -1
#    
#    B = (((A-lim_inf)/float(lim_sup-lim_inf))*(new_sup-new_inf)+new_inf)
#    return B
#    
##From [-1..0..1]  float64 to [0...128...255] uint8  
#def ConvertBalancedTo255(A):  
#    #Masking Saturation    
#    A[A>1] = 1
#    A[A<-1] = -1
#    
#    lim_sup = 1
#    lim_inf = -1
#       
#    new_sup = 255
#    new_inf = 0
#    
#    B = np.round(((A-lim_inf)/float(lim_sup-lim_inf))*(new_sup-new_inf)+new_inf)
#    B = np.uint8(B)
#    return B  
    
    
    
    
    