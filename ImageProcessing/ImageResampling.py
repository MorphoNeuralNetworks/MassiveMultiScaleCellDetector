# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 18:25:13 2020

@author: pc
"""

from scipy import signal
import numpy as np
import time

from ImageProcessing.IntensityMapping import change_ImageIntensityMap
#==============================================================================
# 
#==============================================================================
def resample_Zdim(imgIn, z_thick, dissectionSize_iso, overlap_iso, t0=0):
    start = time.time()-t0
    #Forcing a 3D-Image to be Isotropic through a Resampling Operator    
    if z_thick>1.0:   
        zDim = imgIn.shape[2]*z_thick        
        Nz = (zDim-overlap_iso[0])/(dissectionSize_iso[0]-overlap_iso[0])
        Nz = int(np.round(Nz))
        zDim = Nz*(dissectionSize_iso[0]-overlap_iso[0]) + overlap_iso[0]    
        zDim = int(zDim)
        
        imgIn = signal.resample(imgIn, zDim, axis=2)
    
    stop = time.time()-t0
    return imgIn, start, stop

def resample_3DImage(imgIn, scannerSize_Out, t0=0, verbose=False):
    start = time.time() - t0
    if verbose==True:
        print()
        print('Start: resample_3DImage()')
        print('scannerSize_Out \n', scannerSize_Out)
        ny, nx, nz = imgIn.shape
        print('imgDimXYZ \n', [nx, ny, nz] )
        
    #Determine the Bit Depth
    dataType = imgIn.dtype.type
    bitDepth = 0
    if dataType==np.uint8:
        bitDepth = 8
    elif dataType==np.uint16:
        bitDepth = 16
    else:
        print()
        print('BitDepth not Found')
        print('dataType:', dataType)
        
    print()    
    print('bitDepth', bitDepth)
    print('dataType:', dataType)
    
    print()
    print('Before: Resampled')
    print('imgIn.min()', imgIn.min())
    print('imgIn.max()', imgIn.max())
    
    #yxz
    imgIn = signal.resample(imgIn, scannerSize_Out[0], axis=1) # ??? IMP
    imgIn = signal.resample(imgIn, scannerSize_Out[1], axis=0) # ??? IMP
    imgIn = signal.resample(imgIn, scannerSize_Out[2], axis=2)
    
    print()
    print('After: Resampled')
    print('imgIn.min()', imgIn.min())
    print('imgIn.max()', imgIn.max())
    
    
    # if scannerSize_Out[2]!=0:
    #     imgIn = signal.resample(imgIn, scannerSize_Out[2], axis=2)
    
    # Recompute DataType
    #op1
    # imgIn[imgIn<0] = 0 
    # imgIn[imgIn>(2**bitDepth - 1)] = 2**bitDepth - 1
    # imgIn = imgIn.astype(dataType)
    
    #Op2
    imgIn = change_ImageIntensityMap(imgIn, x0=imgIn.min(), x1=imgIn.max(), y0=0, y1=2**bitDepth-1)    
    imgIn = imgIn.astype(dataType)
    
    print()
    print('After: DataType Restorage')
    print('imgIn.min()', imgIn.min())
    print('imgIn.max()', imgIn.max())
    
    if verbose==True:
        ny, nx, nz = imgIn.shape
        print('imgDimXYZ \n', [nx, ny, nz] )
        print('Stop: resample_3DImage()')
        print()

    stop = time.time()-t0
    return imgIn, start, stop
 




# =============================================================================
# Test       
# =============================================================================
if __name__== '__main__':
    from IO.Image.ImageReader import read_Image
        
    pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\ImageDataSets\Paulina\LSM980\Control'    
    voxelSize_In_um  = 1*np.asarray([0.59, 0.59, 6.0])
    voxelSize_Out_um = 2*np.asarray([1.0, 1.0, 1.0]) 
    
    # pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\MiniTest'
    # voxelSize_In_um  = np.asarray([0.45, 0.45, 2.00])
    # voxelSize_Out_um = 2*np.asarray([1.0, 1.0, 1.0])   
            
    # Read ImgIn
    img3D = read_Image(pathFolder_ReadImage, nThreads=1)

    # Get the Out/In Ratio 
    r_um = voxelSize_Out_um/voxelSize_In_um
    r_px = 1/r_um
    
    # Compute the Output Dimensions
    imgDimYXZ_In = np.array(img3D.shape)  
    imgDimXYZ_In = imgDimYXZ_In[[1,0,2]]
    imgDimXYZ_Out = np.round(imgDimXYZ_In*r_px).astype(int)
        
    print()
    print('Size')
    print('imgDimXYZ_In: ', imgDimXYZ_In) 
    print('imgDimXYZ_Out:', imgDimXYZ_Out) 
    print('r_um:', r_um)
    # print('r_px:', r_px)
    
    #Test
    # a = np.array([4.44444444, 4.44444444, 1.        ])
    # a = np.array([1., 1., 1.        ])
    # (a[0]!=1)&(a[1]!=1)&(a[2]!=1)
    # (a[0]!=1)|(a[1]!=1)|(a[2]!=1)
    
    # Resample Image
    if (r_um[0]!=1)|(r_um[1]!=1)|(r_um[2]!=1): 
        print()
        print('Compute Resampling...')
        [img3D, start, stop]  = resample_3DImage(img3D, imgDimXYZ_Out)
        
    pass