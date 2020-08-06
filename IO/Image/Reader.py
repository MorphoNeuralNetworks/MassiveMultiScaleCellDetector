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


def read_ImagePatch(rootPath, coordinates, dissectionSize):
    x,   y,  z = coordinates
    dx, dy, dz = dissectionSize
    
    #Get Image Paths from the RootFolder
    imgPaths = get_ImagePaths(rootPath)
    
    #Get Image Format
    img = cv2.imread(imgPaths[0], -1)
    dataType = type(img[0,0])
    
    #Get 3D Image Dimensions
    [ny, nx, nz] = get_DimensionsFrom3DImageSequence(imgPaths)

    #Get Extreme Index to extract the Image Patch from the Whole Image
    x0, x1 = get_CenteredExtremes(x, dx)
    y0, y1 = get_CenteredExtremes(y, dy)
    z0, z1 = get_CenteredExtremes(z, dz)
    
#    #Get Dimensions
#    print('diff')
#    print(x1-x0 + 1)
    
    #Initialize an empty Odd 3D Matrix to be filled with the Image Patch    
    imgPatch = get_zerosOddMatrix(dy, dx, dz, dataType)

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

def get_zerosOddMatrix(ny, nx, nz, dataType):
    nx = get_Odd(nx)
    ny = get_Odd(ny)
    nz = get_Odd(nz)
    
    M = np.zeros((ny,nx,nz), dtype=dataType)
    return M
 
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
        

#==============================================================================
# 
#==============================================================================

def read_MergedImage(rootPathImg, v_xyz_ani, dissectionSize_ani):
    
    #Get    
    xyz_origin = v_xyz_ani[0]  - (dissectionSize_ani - 1)/2 
    xyz_final  = v_xyz_ani[-1]  + (dissectionSize_ani - 1)/2 
    dissectionSize = xyz_final - xyz_origin
    xyz_center = xyz_origin + dissectionSize//2
    
    #Read image
    imgIn = read_ImagePatch(rootPathImg, xyz_center, dissectionSize)

    return imgIn


if __name__== '__main__':
  
  
#==============================================================================
#   #Input
#==============================================================================
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\620x905x1708_2tiff_8bit'
    x, y, z = 309, 327, 850
    
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\2482x3620x1708_2tiff_8bit' 
    BrainRegion = 'mCA1'   
    x, y, z = 1238, 1310, 850

    n = 50
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
    


    
    


#==============================================================================
# Draft
#==============================================================================

#==============================================================================
#     Save Slide
#==============================================================================
#    #Set the filePath to save    
#    locaPath = os.path.dirname(sys.argv[0])
#    locaPath = 'D:\\MyPythonPosDoc\\P4_Cell_Detection'
#    saveFolder = 'TestData'
#    fileName = 'img2D_0.tif'
#    savePath = os.path.join(locaPath, saveFolder, fileName)   
#    
#    #Change from float to uint8
#    imgMiddleSlice_8bit = imgMiddleSlice.astype(np.uint8)
#    
#    #Save the image
#    isSaved = cv2.imwrite(savePath, imgMiddleSlice_8bit)
#    print(isSaved)
    
#==============================================================================
#   save 3D numpy Array  
#==============================================================================
#    #Set the filePath to save    
#    locaPath = os.path.dirname(sys.argv[0])
#    locaPath = 'D:\\MyPythonPosDoc\\P4_Cell_Detection'
#    saveFolder = 'TestData'
#    fileName = 'img3D_0.npy'
#    savePath = os.path.join(locaPath, saveFolder, fileName)
#    
#    isSaved = np.save(savePath, img) 
#    print(isSaved)
 
 
#==============================================================================
#  Draft: Permutation
#==============================================================================
#def get_permutationWithRepetition(v):
#    comb = np.array(np.meshgrid(v)).T.reshape(-1, len(v)) 
#    return comb
#    
#    
#    a = np.asarray([1,2,3])    
#    b = np.asarray([1,2]) 
#    x, y = np.meshgrid(a,b)
#    np.dstack((x,y))
#    v = [a,b]
#
#    np.array(np.meshgrid([1, 2, 3], [4, 5], [6, 7])).T.reshape(-1,3)
#    t = np.array(np.meshgrid(a, b)).T.reshape(-1,2)
#    t = np.array(np.meshgrid(a, b, b)).T.reshape(-1,3)
#    np.array(np.meshgrid(a, b)).T
#    np.array(np.meshgrid(v)).T.reshape(-1,2)
#    np.meshgrid(*v)
#    v = [a,b]
#    v = [a,b,b]
#    np.array(np.meshgrid(*v)).T.reshape(-1, len(v)) 
    
    
    
        