# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:47:37 2020

@author: pc
"""


import numpy as np
import cv2
import os

#from ImageFilters import get_DoG

def convolve_2D(data, Fx, Fy):    
    imgOut = cv2.filter2D(data, -1, Fy) 
    
    imgOut = cv2.filter2D(np.transpose(imgOut, axes=(1,0)), -1, Fx)
    imgOut = np.transpose(imgOut, axes=(1,0))   
    
    return imgOut
    
def convolve_3D_3x1D(data, Fx, Fy, Fz):
    
    imgOut = cv2.filter2D(data, -1, Fy, borderType=0) 
    
    imgOut = cv2.filter2D(np.transpose(imgOut, axes=(1,0,2)), -1, Fx, borderType=0)
    imgOut = np.transpose(imgOut, axes=(1,0,2))

    imgOut = cv2.filter2D(np.transpose(imgOut, axes=(2,1,0)), -1, Fz, borderType=0)
    imgOut = np.transpose(imgOut, axes=(2,1,0))
    
    
    return imgOut
    
def convolve_3D_2x2D(data, Fxy, Fyz):
    
    imgOut = cv2.filter2D(data, -1, Fxy)     

    imgOut = cv2.filter2D(np.transpose(imgOut, axes=(2,1,0)), -1, Fyz)
    imgOut = np.transpose(imgOut, axes=(2,1,0))
    
#    imgOut = np.sqrt(2.0)*imgOut
    
    return imgOut

if __name__== '__main__':
    
    #Set the filePath to Read 3D numpy Array    
    locaPath = 'D:\\MyPythonPosDoc\\P4_Cell_Detection'
    saveFolder = 'TestData'
    fileName = 'img3D_0.npy'
    readPath = os.path.join(locaPath, saveFolder, fileName)    
    imgIn = np.load(readPath) 
    
#==============================================================================
#     
#==============================================================================
#    sc = 3.0
#    scX, scY, scZ = 1.5, 1.5, 1.5  
#    Fx = get_DoG(sc=scX, nDim=1)    
#    Fy = get_DoG(sc=scY, nDim=1)
#    Fz = get_DoG(sc=scZ, nDim=1)
#    imgOut = convolve_3D(imgIn, Fx, Fy, Fz)
    
    














#==============================================================================
#     Draft
#==============================================================================
    
      
#import cv2
#import scipy.signal as signal
#import numpy as np
#
#image = np.random.randint(255, size=(5, 5))
#kernel = cv2.getGaussianKernel(13, 2)
#kernel_2D = np.outer(kernel, kernel)
#
#result1 = signal.convolve(image, kernel_2D, mode='same')
#result2 = signal.convolve(signal.convolve(image, kernel, mode='same'), kernel, mode='same')
#
#result3 = cv2.filter2D(image,-1, kernel_2D, borderType=0)
#result4 = cv2.sepFilter2D(image*1.0, -1, kernel, kernel, borderType=0)

##Correction
#result2 = signal.convolve(signal.convolve(image, kernel, mode='same'), kernel.T, mode='same')

    
    
    
    
    
    
    
    
    
    
#==============================================================================
#     Draft
#==============================================================================
    
    
#==============================================================================
#     
#==============================================================================
#    imgOut_y = cv2.filter2D(img, -1, F) 
#    
#    imgOut_x = cv2.filter2D(np.transpose(img, axes=(1,0,2)), -1, F)
#    imgOut_x = np.transpose(imgOut_x, axes=(1,0,2))
#
#    imgOut_z = cv2.filter2D(np.transpose(img, axes=(2,1,0)), -1, F)
#    imgOut_z = np.transpose(imgOut_z, axes=(2,1,0))
    
#==============================================================================
#     
#==============================================================================
#    imgOut = (imgOut_y + imgOut_x + imgOut_z)/3.0
#    imgOut = imgOut_y*imgOut_x*imgOut_z
#    imgOut = imgOut_y*imgOut_x
#    imgOut = imgOut_z
#    imgOut = imgOut_y
#    imgOut = np.sqrt(imgOut_y**2 + imgOut_x**2 + imgOut_z**2)
    
    