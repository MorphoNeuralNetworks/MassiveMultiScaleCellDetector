# -*- coding: utf-8 -*-
"""
Created on Wed May 06 23:06:29 2020

@author: pc
"""


import numpy as np
import pandas as pd
import time



import cv2

from scipy import signal


import os
import sys
from pathlib import Path 


from matplotlib import pyplot as plt
import matplotlib.cm as cm
#import matplotlib
#from mpl_toolkits.mplot3d import Axes3D


from ImageProcessing.ImageFilters import get_Gaussian

from ImageProcessing.IntensityMapping import change_ImageIntensityMap
from IO.Image.Reader import read_ImagePatch
from ImageProcessing.PatchExtractor import get_ImagePatch

#from Plots.GeneralPlots import plot_img1D, plot_img2D
from ImageProcessing.ImageMasks import get_Ellipsoid
from Plots.MultiScale import (plot_MultiScaleAnalysis,
                              plot_2DResult,
                              plot_3DResults,
                              plot_DetectedCells,
                              plot3D_2DCrossSection,
                              save_Results,
                              plot_2DResultTensor)


from IO.Files.Manager import createFolder
from IO.Image.Writer import save3Dimage_as2DSeries

from ImageProcessing.CellLocalizer import (run_CellLocalizer,
#                                                compute_SecondDerivativeMS,
#                                                compute_SpatialMaxMS,
#                                                compute_ScaleMaxMS,
#                                                compute_ScaleCount,
                                                remove_FalseCells_with_LowIntensity,
                                                remove_FalseCells_with_LowScaleCount)

from ImageProcessing.Tensor import run_Tensor
     
      
if __name__== '__main__':
    

#==============================================================================
#   Input Metada
#==============================================================================
    #Metadata:
    r_min_um =  5.0 #Smallest radius of a neuron in micrometers
    r_max_um = 14.0 #Biggest radius of a neuron in micrometers


#==============================================================================
#   Brain Regions
#==============================================================================
    #CA1 
    BrainRegion = 'mCA1'   
    x, y, z = 1238, 1310, 850
    
#    #Axon Bundles
#    BrainRegion = 'mAxonBundles' 
#    x, y, z = 1098, 1377, 850 
#    
    #Subiculum
    BrainRegion = 'mSub_On' 
    x, y, z = 1127, 518, 850 
       
    #Dentate Gyrus
#    BrainRegion = 'mDG_High' 
#    x, y, z = 1723, 864, 850  

#==============================================================================
# Special Cases    
#==============================================================================

#    #Dentate Gyrus
#    BrainRegion = 'mDG_Low' 
#    x, y, z = 1789, 844, 850 
# 
#    #Subiculum
#    BrainRegion = 'mSub_Off' 
#    x, y, z = 1145, 502, 850 
    
#==============================================================================
#   #Get a 3D Patch from a Big 3D Image (High Resolution)
#   #Only supported for Image Sequence given in tif format
#  1.19x1.19x5.00 μm    
#==============================================================================
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\2482x3620x1708_2tiff_8bit' 
    resX, resY, resZ = 1.19, 1.19, 5.00 #Voxel Resoluion in micrometer units 
#    resX, resY, resZ = 1.29, 1.29, 5.00
    z_thick = float(resZ/resX)
    #ATTENTION: Think about this
    z_thick = z_thick/2.0
    
#    n = 101
#    n = 73
    n = 51
    n = 41
#    n = 21
    if n < z_thick:
        n = z_thick

    Nx, Ny = n, n
    Nz = int(np.round(n/z_thick))    
    dx, dy, dz = Nx, Ny, Nz 


#==============================================================================
#   #Get a 3D Patch from a Big 3D Image (low Resolution)
#   #Only supported for Image Sequence given in tif format
#  1.19x1.19x5.00 μm 
#============================================================================== 
#    rootPath = 'D:\\MyPythonPosDoc\\Brains\\620x905x1708_2tiff_8bit'  
#    resX, resY, resZ = 5.00, 5.00, 5.00 #Voxel Resoluion in micrometer units   
#    z_thick = float(resZ/resX)
#    
#    x , y = int(x/4.0), int(y/4.0)
#    
#    n = int(n/4.0)
#    if n < z_thick:
#        n = z_thick
#    Nx, Ny = n, n
#    Nz = int(np.round(n/z_thick))    
#    dx, dy, dz = Nx, Ny, Nz 
#    
    
#==============================================================================
#    Intensity Mapping
#==============================================================================
    imgIn = read_ImagePatch(rootPath, x, y, z, dx, dy, dz)
    imgInAux = imgIn.copy()
#==============================================================================
#     Ressampling: Upsampling (forcing Isotropic 3D Volume)
#==============================================================================
    if z_thick>1.0:
        print('')
        print('Correcting Z-Anistropy...')
        print(imgIn.shape)
        imgIn = signal.resample(imgIn, imgIn.shape[0], axis=2)
        print(imgIn.shape)
        
#==============================================================================
#     #Intenstity Mapping: [0...128...255] -> [-1...0...+1]
#==============================================================================
    imgIn = change_ImageIntensityMap(imgIn, x0=0, x1=255, y0=-1, y1=+1)
    
#==============================================================================
#     #Center the Dynamic Range (subtracting the mean luminance)
#==============================================================================
    imgIn = imgIn - imgIn.mean()
#    jajaj
#    imgIn = change_ImageIntensityMap(imgIn, x0=0, x1=255, y0=0, y1=+1)
#==============================================================================
#   Off Channel    
#==============================================================================
    #On/Of Channels 
#    imgIn = -1*imgIn #On Vs Off
        

#==============================================================================
#   Ellispsoid Model
#==============================================================================    
#    BrainRegion = 'mEllipsoid' 
#    Rx, Ry, Rz = 5, 5, 5    #Sphere
#    Rx, Ry, Rz = 11, 11, 5    #Disk
#    Rx, Ry, Rz = 5, 5, 11    #Tube
#    az, ay, ax = 0, 0, 0
#    R = [Rx, Ry, Rz]
#    A = [ax, ay, az]
#    r = np.min(R)
#    n = 4*np.max(R)
#    
#    imgIn = get_Ellipsoid(R, n=n, A=A, Imin=-1.0, Imax=+1.0)  

#==============================================================================
#   0.0) Ploting: before PreProcessing
#==============================================================================
#    print('')
#    print('Input as Middle Cross Section: Before Smoothing')
#    ny, nx = 1, 3
#    m = 0.75
#    fig, axs = plt.subplots(ny,nx)
#    graphSize = [4.0, 4.0]
#    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
#    fig.set_size_inches(graphSize)
#    
#    plot3D_2DCrossSection(imgIn, axs[:])
#    fig.tight_layout(h_pad=1.0)  
#    plt.show() 

  
#==============================================================================
#   0.1) (Optional) Preproccessing: Smoothing Filter
#   Gauss Smothing (Prefiltering: at the lower scale): 
#==============================================================================
#    R = 3.0    
#    s = [R, R, R]
#    Fxyz = get_Gaussian(s, a=0.25) 
#    imgIn_fft = np.fft.fftn(imgIn)
#    Fxyz_fft =  np.fft.fftn(Fxyz, (imgIn_fft.shape))
#    imgOut_fft = imgIn_fft*np.abs(Fxyz_fft)
#    imgIn =     np.fft.ifftn(imgOut_fft)
#    imgIn = imgIn.real



#==============================================================================
#   0.2) (Optional) Preproccessing: Noise Model
#   Additive White Noise 
#==============================================================================
#    ny, nx, nz = imgIn.shape
#    
#    Ae = 1.0
#    Noise1D = (Ae*np.random.uniform(0,1,(ny, nx, nz)) - Ae/2.0)
#    imgIn = imgIn + Noise1D
#    plt.imshow(imgIn[:, :,nz//2], cm.Greys_r, interpolation='nearest')
#    plt.show()
#    print(Noise1D.min())
#    print(Noise1D.max())
    
      
#==============================================================================
#   0.1) Ploting: After  PreProcessing
#    Plot Cross section of Input Image -> this serve to realize how sensitive 
#   the algorithm is to Low Intensity Values close to the Noise level
#==============================================================================
    print('')
    print('Input as Middle Cross Section: After Smoothing')
    ny, nx = 2, 3
    m = 0.75
    fig, axs = plt.subplots(ny,nx)
    graphSize = [4.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
    fig.set_size_inches(graphSize)

    print('All Dynamic Range Vs Restricted Dynamic Range')
    plot3D_2DCrossSection(imgIn, axs[0,:], Imin=-1, Imax=+1)
    plot3D_2DCrossSection(imgIn, axs[1,:])
    fig.tight_layout(h_pad=1.0)  
    plt.show() 
    
#==============================================================================
#   1) Cell Detection Algorithm
#      1.0) Set the Input Parameters of the Algorihtm 
#==============================================================================

    #1) Second Derivative Parameter 
    #1.1) Spatial Scales
    r_min_px = int(np.floor(r_min_um/resX))
    r_max_px = int(np.ceil(r_max_um/resX))      
    scales = np.arange(r_min_px-1, r_max_px + 1 , dtype=np.float)
    scales = np.arange(r_min_px, r_max_px + 1 , dtype=np.float)

#    scales = np.arange(2, 4, dtype=np.float)

    #1.2) DoG filters Parameters
    rS = 1.1 
    rV = 1.0
    
#    rS = 1.1 
#    rV = 0.9
    
#    rS = 3.5  
#    rV = 1.0
    
    
    #2) Maxima Detection Parameter
    intensityThreshold = 0.0

#==============================================================================
#   1) Cell Location Algorithm: xyz-coordinates of the geometric center of the soma
#==============================================================================

    imgDoGMS, df_All, df_Cells1, dt = run_CellLocalizer(imgIn, scales, rS, rV)
    
#==============================================================================
#   2) Remove: False Positive Cells
#============================================================================== 
    #2.a) Remove detected cells with Low Local Intensities   
#    intensityRatio = 0.00 
#    intensityRatio = 0.10
#    intensityRatio = 0.50
#    df_Cells2 = remove_FalseCells_with_LowIntensity(df_Cells1, intensityRatio=intensityRatio)

    #2.b) Remove detected cells with Low Scale Count
    scaleCountThreshold = 4
    df_Cells2 = remove_FalseCells_with_LowScaleCount(df_Cells1, scaleCountThreshold= scaleCountThreshold)


#==============================================================================
#   Selecting the Results to Compute the Tensor Matrix
#==============================================================================
#    df_Cells2 = df_Cells1.copy()

#==============================================================================
#   Tensor Metrics
#==============================================================================
    df_Cells3 = run_Tensor(imgDoGMS, df_Cells2, scales)
    

    

#==============================================================================
#   Selecting the Results to Compute the Tensor Matrix
#==============================================================================
#    df_Cells = df_Cells1.copy()
#    df_Cells = df_Cells2.copy()   
    df_Cells = df_Cells3.copy()   
    
#==============================================================================
#   Results: Visualization
#==============================================================================
    
    n_decimals = 2
    print('')
    print('-------------------------------------------------------')
    print('-----------0) MultiScale Detection Algorithm------------')
    print('-------------------------------------------------------') 
    fig, axs = plot_2DResult(imgIn, df_All)
    fig.tight_layout(h_pad=1.0)  
    plt.show() 
    print('')
    print('N_cells')
    print(df_All.shape[0])    
    print('')
    print('Table 0')
    print(np.round(df_All, n_decimals))
    
    print('')
    print('-------------------------------------------------------')
    print('-----------1) After Cell Detection Algorithm-----------')
    print('-------------------------------------------------------')       
    fig, axs = plot_2DResult(imgIn, df_Cells1)
    fig.tight_layout(h_pad=1.0)  
    plt.show() 
    print('')
    print('N_cells')
    print(df_Cells1.shape[0])    
    print('')
    print('Table 1')
    print(np.round(df_Cells1, n_decimals))

    print('')
    print('-------------------------------------------------------')
    print('----------2) After Intensity Threholding---------------')
    print('-------------------------------------------------------')      
    fig, axs = plot_2DResult(imgIn, df_Cells2)
    fig.tight_layout(h_pad=1.0)  
    plt.show()
    print('')
    print('N_cells')
    print(df_Cells2.shape[0])   
    print('')
    print('Table 2: After Intensity Threshold')
    print(np.round(df_Cells2, n_decimals))

    print('')
    print('-------------------------------------------------------')
    print('----------3) After Tensor Algorithm--------------------')
    print('-------------------------------------------------------')  
    fig, axs = plot_2DResultTensor(imgIn, df_Cells3)
    fig.tight_layout(h_pad=1.0)  
    plt.show() 
    print('')
    print('N_cells')
    print(df_Cells3.shape[0])        
    print('')
    print('Table 3: After Tensor')
    print(np.round(df_Cells3, n_decimals))
    
#    plot_DetectedCells(imgIn, df_Cells1)
#    plot_DetectedCells(imgIn, df_Cells2)
#    plot_DetectedCells(imgIn, df_Cells3)

    
#==============================================================================
#     Remove based on Anisotropy or Tubularity
#==============================================================================
#    df_Cells = df_Cells3
#    maskBool = df_Cells['Tubularity']>10.0
#    maskBool = df_Cells['Anisotropy']>1.0
#    ix = maskBool.index
#    ix = ix[maskBool] 
#    df_Cells = df_Cells.drop(ix)
#
#    fig, axs = plot_2DResultTensor(imgIn, df_Cells)
#    fig.tight_layout(h_pad=1.0)  
#    plt.show()    
#    
#    print('')
#    print('Table 3')
#    print(np.round(df_Cells, 3))
#==============================================================================
#     #Analize: First Partial Derivatives
#============================================================================== 

#    nny, nnx = 2, 3
#    m = 0.75
#    fig, axs = plt.subplots(nny,nnx)
#    graphSize = [4.0, 4.0]
#    graphSize = m*nnx*graphSize[0], m*nny*graphSize[1]    
#    fig.set_size_inches(graphSize)
#    
#    T = TensorMS[0]
#    axs = axs.ravel()
#    n = axs.shape[0]
#    for i in range(0,n):
#        ax = axs[i]
#        Dxyz = T[i]
#        
#        ny, nx, nz = Dxyz.shape
#        Dxy = Dxyz[:,:, nz//2]
#        D_Center = (Dxy[ny//2, nx//2])
#        myText = '{:0.2f}'.format(D_Center)
#        ax.set_title(myText)
#        ax.imshow(Dxy, cmap = cm.Greys_r, interpolation='nearest',vmax=None,vmin=None)
#    fig.tight_layout(h_pad=1.0) 
#    plt.show()



#'{:0.1f}'.format(ss)
#==============================================================================
#==============================================================================
#==============================================================================
# # ------------------------------------------------
#==============================================================================
#==============================================================================
#==============================================================================

#==============================================================================
#  Save 3D-Input Image
#==============================================================================
    
    #Set the rootFolder to save the stacks
    locaPath = Path(os.path.dirname(sys.argv[0]))
    saveFolder = BrainRegion
    folderPath = Path.joinpath(locaPath, 'Results', saveFolder)  
    
    #Create the Rootfolder
    createFolder(str(folderPath))
    
    #Intensity Map
    imgIn = change_ImageIntensityMap(imgIn, x0=-1, x1=+1, y0=0, y1=+255)
    imgIn = imgIn.astype(np.uint8)
    
    #Op1
    #Save 3D-Image as a Series of 2D images
#    save3Dimage_as2DSeries(imgIn, str(folderPath))
    
    #Op2
    #Save 3D-Image as a 3D Stack
    import tifffile
    filePath = Path.joinpath(folderPath, 'm3Dimage.tif')  
    tifffile.imwrite(filePath, imgIn, photometric='minisblack')
#    tifffile.imwrite(filePath, imgInAux.astype(np.uint8), photometric='minisblack')
    
#==============================================================================
#     Save Cells Vaad3d
#    df = df_Cells1.drop('I', 1)
#==============================================================================

    df = pd.DataFrame()
    n = df_Cells.shape[0]
    ny, nx, nz = imgIn.shape
    df['X'] = df_Cells['Z'] 
    df['Y'] = ny-df_Cells['Y'] 
    df['Z'] = df_Cells['X'] 
    
    
    df['R'] = 100*df_Cells['S'] 
    
    df['shape'] = np.zeros(n)
    df['name'] =  df_Cells.index.values
    df['comment'] = n*['0']
    df['cR'] = 255*np.ones(n)
    df['cG'] = 0*np.ones(n)
    df['cB'] = 0*np.ones(n)
    
    df = df.astype(int)
    
    filePath = Path.joinpath(folderPath, 'mcells.csv')  
    filePath = Path.joinpath(folderPath, 'mcells.marker') 
    df.to_csv(str(filePath), sep=',', encoding='utf-8', index=False, header = False)

    


#==============================================================================
# Draft
#==============================================================================
#    print('Time')
#    print(dt)
#    jaja
    
#    df_Cells1['SNR'] = df_Cells1['I']/np.abs(df_Cells1['I0'])





#



#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
# # # # Results
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================



#==============================================================================
#   Settings
#==============================================================================
#    print('')    
#    print('Scale Pixels (radius)')
#    print (scales)
# 
#    print('')    
#    print('Scale Micrometers (radius)')
#    print (scales*resX)    


    


##==============================================================================
##  Save 3D-Input Image
##==============================================================================
#    
#    #Set the rootFolder to save the stacks
#    locaPath = Path(os.path.dirname(sys.argv[0]))
#    saveFolder = BrainRegion
#    folderPath = Path.joinpath(locaPath, 'Results', saveFolder)  
#    
#    #Create the Rootfolder
#    createFolder(str(folderPath))
#    
#    #Intensity Map
#    imgIn = change_ImageIntensityMap(imgIn, x0=-1, x1=+1, y0=0, y1=+255)
#    imgIn = imgIn.astype(np.uint8)
#    
#    #Op1
#    #Save 3D-Image as a Series of 2D images
##    save3Dimage_as2DSeries(imgIn, str(folderPath))
#    
#    #Op2
#    #Save 3D-Image as a 3D Stack
#    import tifffile
#    filePath = Path.joinpath(folderPath, 'm3Dimage.tif')  
#    tifffile.imwrite(filePath, imgIn, photometric='minisblack')
#    
##==============================================================================
##     Save Cells Vaad3d
##==============================================================================
##    df = df_Cells1.drop('I', 1)
#    df = pd.DataFrame()
#    n = df_Cells2.shape[0]
#    ny, nx, nz = imgIn.shape
#    df['X'] = df_Cells2['Z'] 
#    df['Y'] = ny-df_Cells2['Y'] 
#    df['Z'] = df_Cells2['X'] 
#    
#    
#    df['R'] = 100*df_Cells1['S'] 
#    
#    df['shape'] = np.zeros(n)
#    df['name'] = n*['0']
#    df['comment'] = n*['0']
#    df['cR'] = 255*np.ones(n)
#    df['cG'] = 0*np.ones(n)
#    df['cB'] = 0*np.ones(n)
#    
#    df = df.astype(int)
#    
#    filePath = Path.joinpath(folderPath, 'mcells.csv')  
#    filePath = Path.joinpath(folderPath, 'mcells.marker') 
#    df.to_csv(str(filePath), sep=',', encoding='utf-8', index=False, header = False)
#
#    
    
#==============================================================================
#     Saving Results
#==============================================================================

#    save_Results(imgIn, df_Cells1, rootPath='Results', folderPath=BrainRegion, fileName='1')
#    save_Results(imgIn, df_Cells2, rootPath='Results', folderPath=BrainRegion, fileName='2')




#==============================================================================
#  #Traking the Algorithm Performance through Plots
#==============================================================================

    
#    plot_MultiScaleAnalysis(imgIn, scales, imgDoGMS, df_Cells)    
#    plot_DetectedCells(imgIn, df_Cells)
    
#    print('')
#    print('Results as xyzMIPs')
#    plot_2DResult(imgIn, df_Cells)
#    plot_2DResult(imgIn, df_All)
    
#    plot_3DResults(imgIn, df_Cells)
    
#==============================================================================
#   Computing Performance (efficiency) 
#==============================================================================
      
#    print('')
#    print('Computing Performance...')
#    ny, nx, nz = imgIn.shape  
#    ns = scales.shape[0]
#
#    print('')
#    print('Total Time:', dt1)
#    print('Time per Voxel:', dt1/float(ny*nx*nz*ns)) 
#    
#    print('')
#    print('Total Time:', dt)
#    print('Time per Voxel:', dt/float(ny*nx*nz*ns))
    
    


#==============================================================================
#     Draft
#==============================================================================




#   

