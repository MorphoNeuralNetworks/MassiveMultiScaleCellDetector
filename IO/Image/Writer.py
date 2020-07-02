# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:23:50 2020

@author: pc
"""

from pathlib import Path 
import cv2


def save3Dimage_as2DSeries(img3D, folderPath, imgFormat='.tif'):
    folderPath = Path(folderPath)
    ny, nx, nz = img3D.shape
    
    areSaved = []
    for k in range(0, nz):
        img2Dxy = img3D[:,:,k]
        filePath = Path.joinpath(folderPath, str(k+1))  
        isSaved = save2Dimage(img2Dxy, str(filePath))
        areSaved.append(isSaved)

    return all(areSaved)
        
def save2Dimage(img2D, filePath, imgFormat='.tif'):
    filePath = Path(filePath + imgFormat)
    isSaved = cv2.imwrite(str(filePath), img2D)
    return isSaved


#def createFolder(path):
#    #If the Path already exist remove all the content
#    #If the Path not exist create Folder  
##    path  = Path(path)
#    print(path)
#    if os.path.exists(path): 
#        print('Remove Folder Content...')
#        shutil.rmtree(path)
#        
#    #This dealy is required to avoid the following Error
#    #PermissionError: [WinError 5] Access is denied
#    #It seems that the OS requires time to finish the above operation
#    time.sleep(0.000000001)
#    
#    if not os.path.exists(path):
#        print('Create Folder...')
#        os.makedirs(path)

if __name__== '__main__':
  
#==============================================================================
#   #Get a 3D Patch from a Big 3D Image
#   #Only supported for Image Sequence given in tif format
#============================================================================== 
    #Inputs
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\620x905x1708_2tiff_8bit'     
    x, y, z = 309, 327, 850
    n = 21
    dx, dy, dz = n, n, n 

#==============================================================================
#     
#==============================================================================
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\2482x3620x1708_2tiff_8bit' 
    #Somas    
    x, y, z = 309, 327, 850
    
    #Axones
#    x, y, z = 309-45, 327, 850
    
    #Subiculum
#    x, y, z = 1187, 531, 850
  
    #Dentate Gyrus
#    x, y, z = 1950, 704, 817  
#    x, y, z = 487, 176, 817 
    
    n = 51
    Nx, Ny = n, n
    x , y = 4*x, 4*y
    Nz = int(Nx/4.2)
    dx, dy, dz = Nx, Ny, Nz 
    voxel_dim = 1.19, 1.19, 5.00
#==============================================================================
#     
#==============================================================================
    
    
    #Import of the module in paralllel Folder
    
    from Reader import read_ImagePatch
    
    #Get the image Patch
    img3D = read_ImagePatch(rootPath, x, y, z, dx, dy, dz)
    
#==============================================================================
#     Save 3D image as an Image Stack
#==============================================================================
    import os 
    import sys
    
    #Set the rootFolder to save the stacks
    locaPath = Path(os.path.dirname(sys.argv[0]))
    locaPath = locaPath.parent.parent
    saveFolder = 'myStack'    
    folderPath = Path.joinpath(locaPath, 'TempTest', saveFolder)  
    
    #Import of the module in paralllel Folder
    import sys
    sys.path.append('../')
    from Files.Manager import createFolder
    
    #Create the Rootfolder
    createFolder(str(folderPath))
        
    #Save a 3D-Image (3d-numpy array) as a Image Stack (Sequence of Images)
    a = save3Dimage_as2DSeries(img3D, str(folderPath))
    print(a)
    
    

#==============================================================================
# Draft
#==============================================================================
#    data = [False, False]
#    all(data)
#    any(data)
#    not any(data)
#
#    data = [False, True]
#    all(data)
#    any(data)
#    not any(data)  
#    
#    data = [True, True]
#    all(data)
#    any(data)
#    not any(data) 
    
    
#    fileName = ''
#    savePath = os.path.join(locaPath, saveFolder, fileName)    
#    str(locaPath)

   
   
   
   
   
    