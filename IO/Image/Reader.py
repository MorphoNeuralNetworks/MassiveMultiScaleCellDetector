# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:40:12 2020

@author: pc
"""
#Maths Operation
import numpy as np

#Path Handeling
from glob import glob

#Image Library
import cv2

#Ploting Library
from matplotlib import pyplot as plt
import matplotlib.cm as cm

#operative Systems Libraries       
import os
import sys


#Observation
#1) Only supported for 3D Image Sequence store in the same RootFolder
#2) Only supported for image with '.tif'


def read_ImagePatch(rootPath, x, y, z, dx, dy, dz):
    
    #Get Image Paths from the RootFolder
    imgPaths = get_ImagePaths(rootPath)
    
    #Get 3D Image Dimensions
    [ny, nx, nz] = get_DimensionsFrom3DImageSequence(imgPaths)
    
    #Initialize an empty Odd 3D Matrix to be filled with the Image Patch    
    imgPatch = get_zerosOddMatrix(dy, dx, dz)

    #Get Extreme Index to extract the Image Patch from the Whole Image
    x0, x1 = get_CenteredExtremes(x, dx)
    y0, y1 = get_CenteredExtremes(y, dy)
    z0, z1 = get_CenteredExtremes(z, dz)

    #Get 3D image Patch from the Big 3D Image Sequence
    vz = range(z0, z1+1)    
    for i in range(0, len(vz)):
        imgSlice = cv2.imread(imgPaths[vz[i]], -1)
        imgPatch[:,:,i] = imgSlice[y0:y1+1, x0:x1+1]
    
    return imgPatch
    
def get_ImagePaths(rootPath, imgFormat='tif'):
    nameFilter = '\\*.' + imgFormat
    imgPaths = (glob(rootPath + nameFilter))
    return imgPaths

def get_DimensionsFrom3DImageSequence(imgPaths):
    img = cv2.imread(imgPaths[0], -1)
    ny, nx = img.shape
    nz = len(imgPaths)
    return ny, nx, nz

def get_zerosOddMatrix(ny, nx, nz):
    nx = get_Odd(nx)
    ny = get_Odd(ny)
    nz = get_Odd(nz)
    
    M = np.zeros((ny,nx,nz))
    return M
 
def get_CenteredExtremes(r, dr):
    dr_half = dr//2    
    [r0, r1 ] = [r - dr_half, r + dr_half]
    return r0, r1

def get_Odd(num):
    if (num % 2) == 0: 
        num = num + 1
    return num
        
if __name__== '__main__':
    
#==============================================================================
#   #Input
#==============================================================================
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\620x905x1708_2tiff_8bit'     
    x, y, z = 309, 327, 850
    n = 100
    dx, dy, dz = n, n, n  
#    dx, dy, dz = 31, 20, n 
    
#==============================================================================
#   #Get a 3D Patch from a Big 3D Image
#   #Only supported for Image Sequence given in tif format
#==============================================================================  
    img = read_ImagePatch(rootPath, x, y, z, dx, dy, dz)
#==============================================================================
#     
#==============================================================================
    #Get Image Paths from the RootFolder
    imgPaths = get_ImagePaths(rootPath)
    
    #Get 3D Image Dimensions
    imgWholeShape = get_DimensionsFrom3DImageSequence(imgPaths)  
    
    #Get Patch Dimensions
    imgPatchShape = img.shape
    
    print('WholeDimensions=', imgWholeShape)
    print('PatchDimensions=', imgPatchShape)

#==============================================================================
#   Plotting the Image
#============================================================================== 
    imgMiddleSlice = img[:,:,dz//2]    
    plt.imshow(imgMiddleSlice,  cm.Greys_r, interpolation='nearest') 
    plt.show()
    
    jjjaa

#==============================================================================
#     Save Slide
#==============================================================================
    #Set the filePath to save    
    locaPath = os.path.dirname(sys.argv[0])
    locaPath = 'D:\\MyPythonPosDoc\\P4_Cell_Detection'
    saveFolder = 'TestData'
    fileName = 'img2D_0.tif'
    savePath = os.path.join(locaPath, saveFolder, fileName)   
    
    #Change from float to uint8
    imgMiddleSlice_8bit = imgMiddleSlice.astype(np.uint8)
    
    #Save the image
    isSaved = cv2.imwrite(savePath, imgMiddleSlice_8bit)
    print(isSaved)
    
#==============================================================================
#   save 3D numpy Array  
#==============================================================================
    #Set the filePath to save    
    locaPath = os.path.dirname(sys.argv[0])
    locaPath = 'D:\\MyPythonPosDoc\\P4_Cell_Detection'
    saveFolder = 'TestData'
    fileName = 'img3D_0.npy'
    savePath = os.path.join(locaPath, saveFolder, fileName)
    
    isSaved = np.save(savePath, img) 
    print(isSaved)
    
    
    
    
    
    
    
        