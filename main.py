# -*- coding: utf-8 -*-
"""
Created on Wed May 06 23:06:29 2020

@author: pc
"""


import numpy as np
import pandas as pd
import time



#import cv2
#from scipy import signal


import os
import sys
from pathlib import Path 


from matplotlib import pyplot as plt
import matplotlib.cm as cm
#import matplotlib
#from mpl_toolkits.mplot3d import Axes3D


#from ImageProcessing.ImageFilters import get_Gaussian

from ImageProcessing.IntensityMapping import change_ImageIntensityMap
from IO.Image.Reader import read_ImagePatch, read_MergedImage
#from ImageProcessing.PatchExtractor import get_ImagePatch

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
from IO.Files.Writer import (save_CSV, 
                             save_Vaa3DMarker,
                             save_Vaa3DMarker_Abs,
                             save_Figure
                             )

from IO.Image.Writer import save3Dimage_as3DStack

from ImageProcessing.CellLocalizer import (run_CellLocalizer,
                                           get_spatialScaleRangeInPixels,
                                           remove_FalseCells_with_LowIntensity,
                                           remove_FalseCells_with_LowScaleCount)

from ImageProcessing.Tensor import run_Tensor
from IO.Image.Scanner import (get_scanningDistance, get_scanningLocations)    


from TableProcessing.OverlapManager import (get_centralMask, 
                                            get_overlapMask
                                            )
      
from TableProcessing.TableManager import (get_xyzString,
                                          add_overlapInformation,
                                          add_referenceCoordinates,
                                          add_absoluteCoordinates,
                                          merge_Tables,
                                          remove_MultiDetectionInOverlapedRegions
                                          )
                                          
from IO.Files.Reader import get_pathFiles

from ImageProcessing.ImageResampling import resample_Zdim


if __name__== '__main__':
    
    
#==============================================================================
#   Brain Regions
#==============================================================================
    #CA1 
    BrainRegion = 'mCA1'   
    x, y, z = 1238, 1310, 850
    x, y, z = 1238, 1310 - 25, 850 - 22 #corner
#    x, y, z = 1238, 1335,  828 #cells to close
#    x, y, z = 1251, 1303, int(1787/2.1)  #axon
#    x, y, z = 1282, 1353, int(1842/2.1)  #bifurcation


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
#     
#==============================================================================
  
  

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
    intensityThreshold = 0.15

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
    if (nz % 2) == 0: 
        nz = nz + 1
        
    dissectionSize_ani = np.asarray([nx, ny, nz])
    dissectionSize_iso = np.asarray([nx, ny, nx])

    #Start Point (centered) 
    x0, y0, z0 = x, y, z 
    xyz_start = np.asarray([x0, y0, z0])
    
    #End Point (centered) 
    x1, y1, z1 = x0 + 2*nx, y0 + 0*ny, z0 + 0*nz
#    x1, y1, z1 = x0 + 0*nx, y0 + 0*ny, z0 + 2*nz
#    x1, y1, z1 = x0 + 2*nx, y0 + 2*ny, z0 + 2*nz
    x1, y1, z1 = x0 + 2*nx, y0 + 2*ny, z0 + 2*nz
    xyz_end = np.asarray([x1, y1, z1])


    
    #Overlap (as pixels)
    #Note: the overalp should be the radius of the bigger neuron
    mode = 'pixels'
    r_max = np.max(scales)
    r_max = 21
    overlap_ani = np.asarray([r_max, r_max, np.round(r_max/z_thick)])
    overlap_iso = np.asarray([r_max, r_max, r_max])
 
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
    folderPathTemp = os.path.join(folderPath, 'myTemp')
    

#==============================================================================
# Main
#==============================================================================
    
    #Create the folderPath to Store the Results    
    createFolder(str(folderPath))
    createFolder(str(folderPathTemp))
    
    #Get the scanning coordinates of the Asitropy
    d = get_scanningDistance(dissectionSize_ani, overlap_ani, mode=mode)
    v_xyz_ani = get_scanningLocations(xyz_start, xyz_end, d)    
    
    
    #Get the scanning coordinates if it were isotropic
    v0_iso = np.asarray([xyz_start[0], xyz_start[1], int(z_thick*xyz_start[2])])
    v1_iso = np.asarray([xyz_end[0], xyz_end[1], int(z_thick*xyz_end[2])])
    d_iso = np.asarray([d[0], d[1], d[0]])
    v_xyz_iso = get_scanningLocations(v0_iso, v1_iso, d_iso)    
    
    
    
    #Start the Loop for each disected Volume
    nDissectors = v_xyz_ani.shape[0]
    for i in range(0, nDissectors):
        
        #-------------------------------------  
        #--------Get a Dissected Image--------  
        #------------------------------------- 
        
        #Get Position of Sanning Volume
        xyz_center_ani = v_xyz_ani[i].copy()
        xyz_center_iso = v_xyz_iso[i].copy()
        
        #Read Image Patch
        imgIn = read_ImagePatch(rootPathImg, xyz_center_ani, dissectionSize_ani)
        
        #Get Image Dimensions
        [ny, nx, nz] = imgIn.shape
        imgDim_ani = np.asarray([nx, ny, nz])
       
        #Forcing the 3D-Image to be Isotropic through a Resampling Operator
        imgIn = resample_Zdim(imgIn, z_thick, dissectionSize_iso, overlap_iso)
        
        #Update image dimensions after forcing Isotropy
        [ny, nx, nz] = imgIn.shape
        imgDim_iso = np.asarray([nx, ny, nz])        
       
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
        df_Cells1 = remove_FalseCells_with_LowIntensity(df_Cells0, intensityThreshold, mode='absolute')
    
        #2.b) Remove detected cells with Low Scale Count
        df_Cells2 = remove_FalseCells_with_LowScaleCount(df_Cells1, scaleCountThreshold= scaleCountThreshold)
    
        #Bypassing the threshold of n_scales ()
        df_Cells2 = df_Cells1.copy()
    
        #3) Compute Tensor Metrics: Orientation, Anisotropy, Tubularity, Disck
        df_Cells3, dt2 = run_Tensor(imgDoGMS, df_Cells2, scales)
        
        #4) Selecting the Results to Store
        df_Cells = df_Cells3
        
        
        
        #==============================================================================
        #   Add Coordinates Information to the Table (i.e. df_cells) 
        #==============================================================================
        #Get the Corner (upper,left,front corner) as the Origin of the Referece System
        xyz_corner_iso = xyz_center_iso - (imgDim_iso - 1)/2 

        #Add the Origin Coordinate to the current Volume
        df_Cells = add_referenceCoordinates(df_Cells, origin=xyz_corner_iso)
        
        #Add the Absolute Coordinates of the Detections to the current Volume
        df_Cells = add_absoluteCoordinates(df_Cells, origin=xyz_corner_iso)         
            
        
        #==============================================================================
        # Extract Detections from the Central Region separetely from the Overalp Region
        #==============================================================================
        #Get all Detection Coordinates
        xyzCells = df_Cells['X'].values, df_Cells['Y'].values, df_Cells['Z'].values 
        
        #Extract Detections from the Central Region (i.e. Non-Overlaping Region)  
        maskCenter = get_centralMask(xyzCells, xyz_corner_iso, overlap_iso, imgDim_iso)        
        df_central = df_Cells[maskCenter]
        
        #Extract Detections from the Overlap Region 
        maskOverlap = get_overlapMask(xyzCells, xyz_corner_iso, overlap_iso, imgDim_iso)
        df_overlap = df_Cells[maskOverlap]
    
        #Add Overlap Information to the Overlap Table
        df_overlap = add_overlapInformation(df_overlap, xyz_corner_iso, overlap_iso, imgDim_iso)


        #==============================================================================
        #   Save the Results  (required)    
        #==============================================================================


        #Save Table: Central Region
        fileName = get_xyzString(xyz_corner_iso) + '_CentralRegion'
        save_CSV(df_central, folderPathTemp, fileName)
        
        #Save Table: Overlap Region      
        fileName = get_xyzString(xyz_corner_iso) + '_OverlapRegion'
        save_CSV(df_overlap, folderPathTemp, fileName)        
  
        #==============================================================================
        #    Save Debuugin Results   
        #==============================================================================
  
#        #Save Table: Central Region + Overlap Region
#        fileName = get_xyzString(xyz_corner_iso) + '_BothRegion'
#        save_CSV(df_Cells, folderPathTemp, fileName)        
        
        #File Name
        fileName = get_xyzString(xyz_corner_iso)
        
        #Save 3D-Image 
        imgIn8bit = change_ImageIntensityMap(imgIn, x0=-1, x1=+1, y0=0, y1=+255)
        imgIn8bit = imgIn8bit.astype(np.uint8)
        save3Dimage_as3DStack(imgIn8bit, folderPathTemp, fileName)        
        
        #Save Vaa3D Marker File 
        save_Vaa3DMarker(df_Cells, imgDim_iso, folderPathTemp, fileName) 
        
        #Plot Detected Cells
        fig, axs = plot_2DResultTensor(imgIn, df_Cells, overlap_iso)
        fig.tight_layout(h_pad=1.0) 
        fig_title = fileName + ', N_cells=' + str(df_Cells.shape[0])
        axs[1].set_title(fig_title, fontsize=16)
        plt.show()        
        
        #Save the Plot
        save_Figure(fig, folderPathTemp, fileName)
        
        #==============================================================================
        # Visualizing Debugging Plots (InterMediate Results for Debugging)
        #==============================================================================
#        plot_InterMediateResult(imgIn, df_All, df_Cells0, df_Cells1, df_Cells2, df_Cells3, overlap)

        
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
        
#        break


#==============================================================================
#   Merge Tables of both Central Region and Overlap Region
#==============================================================================

    #Merge the Tables of all Volumes
    pathCenterDetections  = get_pathFiles(folderPathTemp, "/*CentralRegion.csv")
    pathOverlapDetections = get_pathFiles(folderPathTemp, "/*OverlapRegion.csv")
    merged_CentralRegion, merged_OverlapRegion = merge_Tables(pathCenterDetections, pathOverlapDetections)
   
    #Save
    fileName = 'merged_CentralRegion'
    save_CSV(merged_CentralRegion, folderPath, fileName)

    #Save
    fileName = 'merged_OverlapRegion_MultiDetections'
    save_CSV(merged_OverlapRegion, folderPath, fileName)
   
#==============================================================================
#   Remove Multiple Detection in the Ovelap Table   
#==============================================================================     
    #Remove Multiple Detection   
    merged_OverlapRegion = remove_MultiDetectionInOverlapedRegions(merged_OverlapRegion, v_xyz_iso, dissectionSize_iso, overlap_iso)   

    #Save
    fileName = 'merged_OverlapRegion_MultiDetectionsRemoved'
    save_CSV(merged_OverlapRegion, folderPath, fileName)
            
#==============================================================================
#   Get the Final Results
#==============================================================================
    #Remove Inecesary Columns from the merged Overlap Table
    merged_OverlapRegion = merged_OverlapRegion.drop(['overlapRef', 'overlapRegion','overlapID'], axis=1) 
     
    #Merge
    mergedTable = pd.concat([merged_CentralRegion, merged_OverlapRegion], ignore_index=True)

    #Save: Final Result
    fileName = 'merged_FinalResult'
    save_CSV(mergedTable, folderPath, fileName)
      

#==============================================================================
#   Save the Total Image for Debuggin Purpose
#==============================================================================
    #Read Total Image
    imgInTotal = read_MergedImage(rootPathImg, v_xyz_ani, dissectionSize_ani)
        
    #Forcing the 3D-Image to be Isotropic through a Resampling Operator
    imgInTotal = resample_Zdim(imgInTotal, z_thick, dissectionSize_iso, overlap_iso)
        
    #Save 3D-Image 
    imgIn8bit = imgInTotal.astype(np.uint8)
    save3Dimage_as3DStack(imgIn8bit, folderPath, fileName)
    
    #Save the Vaa3D marker file for the Total Image     
    [ny, nx, nz] = imgInTotal.shape
    imgDimGlobal = np.asarray([nx, ny, nz] )
    xyz_origin = v_xyz_iso[0]  - (dissectionSize_iso - 1)/2  - 1
    save_Vaa3DMarker_Abs(mergedTable, xyz_origin, imgDimGlobal, folderPath, fileName)


#==============================================================================
# 
#==============================================================================





#df.drop(['overlapRef', 'overlapRegion','overlapID'], axis=1, inplace=True) 
#pd.read_csv(pathOverlapDetections[0], sep=';', encoding='utf-8',  header = 0, index_col = 0)

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