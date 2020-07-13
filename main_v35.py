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
                              plot_2DResultTensor,
                              plot_InterMediateResult)


from IO.Files.Manager import createFolder
from IO.Files.Writer import save_CSV, save_Vaa3DMarker, save_Figure

from IO.Image.Writer import save3Dimage_as3DStack

from ImageProcessing.CellLocalizer import (run_CellLocalizer,
                                           get_spatialScaleRangeInPixels,
#                                                compute_SecondDerivativeMS,
#                                                compute_SpatialMaxMS,
#                                                compute_ScaleMaxMS,
#                                                compute_ScaleCount,
                                                remove_FalseCells_with_LowIntensity,
                                                remove_FalseCells_with_LowScaleCount)

from ImageProcessing.Tensor import run_Tensor

from IO.Image.Scanner import (get_scanningDistance, get_scanningLocations)    
      
if __name__== '__main__':
    
#==============================================================================
#   Brain Regions
#==============================================================================
    #CA1 
    BrainRegion = 'mCA1'   
    x, y, z = 1238, 1310, 850
#    x, y, z = 1267, 1310, 850 
#    x, y, z = 1356, 1310,  850
  
#    #Axon Bundles
#    BrainRegion = 'mAxonBundles' 
#    x, y, z = 1098, 1377, 850 
    
    #Subiculum
#    BrainRegion = 'mSub_On' 
#    x, y, z = 1127, 518, 850 
##    x, y, z = 1229, 518, 850  
#    x, y, z = 1221, 505, 846 #two cell too close->n=1, I_DoG=0.5

       
#    #Dentate Gyrus
#    BrainRegion = 'mDG_High' 
#    x, y, z = 1723, 864, 850  

    #------------------------------   
    #-------Special Cases----------  
    #------------------------------  

#    #Dentate Gyrus
#    BrainRegion = 'mDG_Low' 
#    x, y, z = 1789, 844, 850 
# 
#    #Subiculum
#    BrainRegion = 'mSub_Off' 
#    x, y, z = 1145, 502, 850 
    
   

#==============================================================================
#   Settings of the Alorithmn
#==============================================================================
    
    #-------------------------------------  
    #--------------Metada: Cell Size------  
    #------------------------------------- 
    
    #0) Cell Size Target 
    r_min_um =  5.0 #Smallest radius of a neuron in micrometers
    r_max_um = 14.0 #Biggest radius of a neuron in micrometers

    #-------------------------------------  
    #--------------3D-Image Data----------  
    #------------------------------------- 
    
    #1.1) Root Path of the 3D-Image    
    rootPathImg = 'D:\\MyPythonPosDoc\\Brains\\2482x3620x1708_2tiff_8bit' 
    
    #1.2) Voxel Resoluion in micrometer units 
    resX, resY, resZ = 1.19, 1.19, 5.00 
    resolution = [resX, resY, resZ]
    
    #-------------------------------------  
    #-------Cell Finder Settings----------  
    #------------------------------------- 
     
    #2.1) Spatial Scale  
    scales = get_spatialScaleRangeInPixels(r_min_um, r_max_um, resolution)

    #2.2) Parameters of the DoG Filter (second derivative filter)
    rS, rV = 1.1, 1.0    
#    rS, rV = 2.1, 1.0 
    #Note: try (rS=3.5, rV=1.0) or (rS=1.1, rV=1.0)
  
    #2.3) Intensity Threshold for the Maxima Detection Filter
    intensityThreshold = 0.0

    #2.4) Remove False Positive based on the n-times that the same cell is detected
    scaleCountThreshold = 4
    scaleCountThreshold = 3
    
    #-------------------------------------  
    #--------Dissection Settings----------  
    #------------------------------------- 

    #Size of the Computing Unit
    n = 41 
    n = 71
#    n = 81
#    n = 101 
    
    #Computing Anisotropy in the Z-direction    
    z_thick = float(resZ/resX)
    #ATTENTION: Think about this
    z_thick = z_thick/2.0
    
#    get_dissectionSize(n, z_thick)
    
    if n < z_thick:
        n = z_thick
    nx, ny = n, n
    nz = int(np.round(n/z_thick))    
    dissectionSize = [nx, ny, nz]

    #Start Point (centered) 
    x0, y0, z0 = x, y, z 
    v0 = [x0, y0, z0] 
    
    #End Point (centered) 
    x1, y1, z1 = x0 + 2*nx, y0, z0
    v1 = [x1, y1, z1]

    #Overlap (as pixels)
    #Note: the overalp should be the radius of the bigger neuron
    mode = 'pixels'
    r_max = np.max(scales)
    r_max = 20
    overlap = [r_max, r_max, np.round(r_max/z_thick)]
 
    #-------------------------------------  
    #-------Tensor Settings----------  
    #------------------------------------- 
#    a_First_Derivative = 0.25
#    a_Gaussian = 0.5


    #-------------------------------------  
    #-------Saving Folder----------  
    #------------------------------------- 

    #Save the Results
    localPath = os.path.dirname(sys.argv[0])
    rootName = 'Results'
    folderName = BrainRegion
    folderPath = os.path.join(localPath, rootName, folderName)
    

#==============================================================================
# Main
#==============================================================================
    
    #Create the folderPath to Store the Results
    createFolder(str(folderPath))
    
    #Get the scanning coordinates
    d = get_scanningDistance(dissectionSize, overlap, mode=mode)
    v_xyz = get_scanningLocations(v0, v1, d)  
    
    nDissectors = v_xyz.shape[0]
    
    for i in range(0, nDissectors):
        
        #-------------------------------------  
        #--------Get a Dissected Image--------  
        #------------------------------------- 
        
        #Get Dissection Volume
        coord_center = v_xyz[i]
        imgIn = read_ImagePatch(rootPathImg, coord_center, dissectionSize)
       
        #Forcing the 3D-Image to be Isotropic through a Resampling Operator   
        if z_thick>1.0:        
            imgIn = signal.resample(imgIn, imgIn.shape[0], axis=2)
        
        #Update image dimensions after forcing Isotropy
        [ny, nx, nz] = imgIn.shape
        imgDim = np.asarray([nx, ny, nz])
        
        #Update the overlap after forcing Isotropy
        overlap = np.asarray([r_max, r_max, r_max])
        
         
        #Intenstity Mapping: [0...128...255] -> [-1...0...+1]
        imgIn = change_ImageIntensityMap(imgIn, x0=0, x1=255, y0=-1, y1=+1)
        
 
        
        #Center the Dynamic Range (subtracting the mean luminance)    
        #The z score tells you how many standard deviations from the mean your score is
        imgIn = imgIn -  imgIn.mean()  
#        imgIn = (imgIn -  imgIn.mean() )/imgIn.std()
        
#        print('')
#        print('Statistics:', i)
#        print('Mean', imgIn.mean())  
#        print('Std', imgIn.std())  
        
        #------------------------------------- ------- 
        #--------Run the Detection Algorithm----------  
        #---------------------------------------------
        
        #1) Cell Location Algorithm: xyz-coordinates of the geometric center of the soma
        imgDoGMS, df_All, df_Cells0, dt1 = run_CellLocalizer(imgIn, scales, rS, rV)
        
        #2) Remove: False Positive Cells
        #2.a) Remove detected cells with Low Local Intensities 
        #Note: the threshold is applied over I_DoG (not I0)
        I_th = 0.0
        I_th = 0.15
#        I_th = 0.2
#        I_th = imgIn.std()
#        print('Std', I_th) 
        df_Cells1 = remove_FalseCells_with_LowIntensity(df_Cells0, I_th, mode='absolute')
    
        #2.b) Remove detected cells with Low Scale Count
        df_Cells2 = remove_FalseCells_with_LowScaleCount(df_Cells1, scaleCountThreshold= scaleCountThreshold)
    
        #Debugging: Selecting the Results to Compute the Tensor Matrix
        #Bypassing the threshold on n_scales
#        df_Cells2 = df_Cells1.copy()
    
        #3) Compute Tensor Metrics: Orientation, Anisotropy, Tubularity, Disck
        df_Cells3, dt2 = run_Tensor(imgDoGMS, df_Cells2, scales)
        
    
        #-------------------------------------------- 
        #--------Coordinates Management------------------ 
        #---------------------------------------------      
        
        #Selecting the Results to Store
        df_Cells = df_Cells3
        
        #Absolute Coordinates of the Corner
        #Note: isotropy was assumed-> z_res=x_res
        coord_corner = coord_center - (imgDim - 1)/2       
        df_Cells['X_abs'] = df_Cells['X'] + coord_corner[0]
        df_Cells['Y_abs'] = df_Cells['Y'] + coord_corner[1]
        df_Cells['Z_abs'] = df_Cells['Z'] + coord_corner[2]

        x0, y0, z0 = overlap
        x1, y1, z1 = imgDim - overlap
        v_corner1 = coord_corner + [0*x1, 0*y1, 0*z1 ]
        v_corner2 = coord_corner + [1*x1, 0*y1, 0*z1 ]
        v_corner3 = coord_corner + [0*x1, 1*y1, 0*z1 ]
        v_corner4 = coord_corner + [1*x1, 1*y1, 0*z1 ]
        
        v_corner5 = coord_corner + [0*x1, 0*y1, 1*z1 ]
        v_corner6 = coord_corner + [1*x1, 0*y1, 1*z1 ]
        v_corner7 = coord_corner + [0*x1, 1*y1, 1*z1 ]
        v_corner8 = coord_corner + [1*x1, 1*y1, 1*z1 ]
        
        #Corners
        boolMask = (df_Cells['X'] < x0)*(df_Cells['Y'] < y0)*(df_Cells['Z'] < z0)
        df_corner1 = df_Cells[boolMask]
       
        boolMask = (df_Cells['X'] > x1)*(df_Cells['Y'] < y0)*(df_Cells['Z'] < z0)
        df_corner2 = df_Cells[boolMask]
        
        boolMask = (df_Cells['X'] < x0)*(df_Cells['Y'] > y1)*(df_Cells['Z'] < z0)
        df_corner3 = df_Cells[boolMask]

        boolMask = (df_Cells['X'] > x1)*(df_Cells['Y'] > y1)*(df_Cells['Z'] < z0)
        df_corner4 = df_Cells[boolMask]
        
        boolMask = (df_Cells['X'] < x0)*(df_Cells['Y'] < y0)*(df_Cells['Z'] > z1)
        df_corner5 = df_Cells[boolMask]
       
        boolMask = (df_Cells['X'] > x1)*(df_Cells['Y'] < y0)*(df_Cells['Z'] > z1)
        df_corner6 = df_Cells[boolMask]
        
        boolMask = (df_Cells['X'] < x0)*(df_Cells['Y'] > y1)*(df_Cells['Z'] > z1)
        df_corner7 = df_Cells[boolMask]

        boolMask = (df_Cells['X'] > x1)*(df_Cells['Y'] > y1)*(df_Cells['Z'] > z1)
        df_corner8 = df_Cells[boolMask]        
        
        #vBars
        boolMask = (df_Cells['X'] < x0)*(df_Cells['Y'] > y0)*(df_Cells['Y'] < y1)*(df_Cells['Z'] < z0)
        df_vBar1 = df_Cells[boolMask]
        
        boolMask = (df_Cells['X'] > x1)*(df_Cells['Y'] > y0)*(df_Cells['Y'] < y1)*(df_Cells['Z'] < z0)
        df_vBar2 = df_Cells[boolMask]
 
        boolMask = (df_Cells['X'] < x0)*(df_Cells['Y'] > y0)*(df_Cells['Y'] < y1)*(df_Cells['Z'] > z1)
        df_vBar3 = df_Cells[boolMask]
        
        boolMask = (df_Cells['X'] > x1)*(df_Cells['Y'] > y0)*(df_Cells['Y'] < y1)*(df_Cells['Z'] > z1)
        df_vBar4 = df_Cells[boolMask]
        
        #hBars
        boolMask = (df_Cells['X'] > x0)*(df_Cells['X'] < x1)*(df_Cells['Y'] < y0)*(df_Cells['Z'] < z0)
        df_hBar1 = df_Cells[boolMask]
        
        boolMask = (df_Cells['X'] > x1)*(df_Cells['Y'] > y0)*(df_Cells['Y'] < y1)*(df_Cells['Z'] < z0)
        df_hBar2 = df_Cells[boolMask]
 
        boolMask = (df_Cells['X'] < x0)*(df_Cells['Y'] > y0)*(df_Cells['Y'] < y1)*(df_Cells['Z'] > z1)
        df_hBar3 = df_Cells[boolMask]
        
        boolMask = (df_Cells['X'] > x1)*(df_Cells['Y'] > y0)*(df_Cells['Y'] < y1)*(df_Cells['Z'] > z1)
        df_hBar4 = df_Cells[boolMask]        
        
        
        
        
#        #Boudanries
#        x_left  = coord_corner[0] + overlap[0]
#        x_right = coord_corner[0] + (imgDim[0] - overlap[0])
#        
#        y_up    = coord_corner[1] + overlap[1]
#        y_down  = coord_corner[1] + (imgDim[1] - overlap[1])
#        
#        z_front = coord_corner[2] + overlap[2]
#        z_back  = coord_corner[2] + (imgDim[2] - overlap[2])
#        
#        #Center Area
#        boolMask = ((df_Cells['X_abs'] >= x_left ) * (df_Cells['X_abs'] <= x_right) *
#                    (df_Cells['Y_abs'] >= y_up   ) * (df_Cells['Y_abs'] <= y_down ) *
#                    (df_Cells['Z_abs'] >= z_front) * (df_Cells['Z_abs'] <= z_back )
#                    )                                 
#        df_center = df_Cells[boolMask]
#        
#        #Overlap Areas (same Corner coordinate)                    
#        df_V_left    = df_Cells[df_Cells['X_abs'] < x_left ]
#        df_H_up      = df_Cells[df_Cells['Y_abs'] < y_up   ]
#        df_D_front   = df_Cells[df_Cells['Z_abs'] < z_front]
#        
#        #Overlap Areas (different Corner coordinate) 
#        df_V_right  = df_Cells[df_Cells['X_abs'] > x_right ]
#        df_H_down   = df_Cells[df_Cells['Y_abs'] > y_down  ]
#        df_D_back   = df_Cells[df_Cells['Z_abs'] > z_back  ]
#        
#        print(df_center)
#        print(df_V_left)
#        print(df_H_up)
#        print(df_D_front)
        
        #==============================================================================
        #   #Save the Results      
        #==============================================================================
        #File Name
        x, y, z = coord_corner
        fileName = 'x_' + str(int(x)) + '_y_' + str(int(y)) + '_z_' + str(int(z))
        
        #Save Results Figure
        fig, axs = plot_2DResultTensor(imgIn, df_Cells, overlap)
        fig.tight_layout(h_pad=1.0) 
        fig_title = fileName + ', N_cells=' + str(df_Cells.shape[0])
        axs[1].set_title(fig_title, fontsize=16)
        plt.show() 
        save_Figure(fig, folderPath, fileName)
        
        #Pring Results: Table
        print(np.round(df_Cells, 2))

        #Save Results: Table as CSV
        save_CSV(df_Cells, folderPath, fileName)
        
        #Save Results: Table as Vaa3DMarker 
        save_Vaa3DMarker(df_Cells, imgDim, folderPath, fileName) 
        
        #Save 3D-Image 
        imgIn8bit = change_ImageIntensityMap(imgIn, x0=-1, x1=+1, y0=0, y1=+255)
        imgIn8bit = imgIn8bit.astype(np.uint8)
        save3Dimage_as3DStack(imgIn8bit, folderPath, fileName)
        

        #Plot InterMediate Results for Debugging
#        plot_InterMediateResult(imgIn, df_All, df_Cells0, df_Cells1, df_Cells2, df_Cells3)

        #==============================================================================
        # Computing Efficiency
        #==============================================================================
        t = np.sum(dt1) + np.sum(dt2)
        v = float(n)**3
        Nx, Ny, Nz = 2482, 3620, 1708
        V = float(Nx*Ny*Nz*z_thick)
        T = V/v*t
        
        print('')
        print('dt1=',dt1)
        print('dt2=',dt2)
        print('t=',t)
        
        print('')
        print('ms per voxel', t/v*10**6)        
        
        print('')
        print('Total:')
        print('Secs =', T)
        print('Mins =', T/(60.))
        print('Hours=', T/(60.*60.))
        print('Days =', T/(60.*60.*24))
        
        break

#==============================================================================
#    Analysis: Computing Time
#==============================================================================
    #Obeservation:
    #the cell detection algorithm is faster when computing 
    #the second derivative of the image through a convolution
    #in the space domain instead of in the frequency domain.
    #spatial domain:    4 secs when 71x71x71 image
    #frequency domain: 10 secs when 71x71x71 image


#==============================================================================
#   Merged
#==============================================================================






#==============================================================================
#==============================================================================
#==============================================================================
# # ------------------------------------------------
#==============================================================================
#==============================================================================
#==============================================================================
#        plot_DetectedCells(imgIn, df_Cells1)
#        plot_DetectedCells(imgIn, df_Cells2)
#        plot_DetectedCells(imgIn, df_Cells3)

        #==============================================================================
        #   Input: Visualization (Plot Cross section of Input Image)
        #==============================================================================
#        print('')
#        print('Input as Middle Cross Section: After Smoothing')
#        nny, nnx = 2, 3
#        m = 0.75
#        fig, axs = plt.subplots(nny,nnx)
#        graphSize = [4.0, 4.0]
#        graphSize = m*nnx*graphSize[0], m*nny*graphSize[1]    
#        fig.set_size_inches(graphSize)
#    
#        print('All Dynamic Range Vs Restricted Dynamic Range')
#        plot3D_2DCrossSection(imgIn, axs[0,:], Imin=-1, Imax=+1)
#        plot3D_2DCrossSection(imgIn, axs[1,:])
#        fig.tight_layout(h_pad=1.0)  
#        plt.show() 

    
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
#   #Get a 3D Patch from a Big 3D Image (High Resolution)
#   #Only supported for Image Sequence given in tif format
#  1.19x1.19x5.00 μm    
#==============================================================================
#    rootPath = 'D:\\MyPythonPosDoc\\Brains\\2482x3620x1708_2tiff_8bit' 
#    #Voxel Resoluion in micrometer units 
#    resX, resY, resZ = 1.19, 1.19, 5.00 
#    resolution = [resX, resY, resZ]
#    resX, resY, resZ = 1.29, 1.29, 5.00
    
    
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