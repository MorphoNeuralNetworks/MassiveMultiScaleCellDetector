# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 19:58:10 2020

@author: pc
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import os
import sys
import psutil
import threading


from ImageProcessing.ImageResampling import resample_Zdim, resample_3DImage
from ImageProcessing.IntensityMapping import change_ImageIntensityMap

from ImageProcessing.CellLocalizer import (run_CellLocalizer,
                                           get_spatialScaleRangeInPixels,
                                           remove_FalseCells_with_LowIntensity,
                                           remove_FalseCells_with_LowScaleCount)
                                           
from ImageProcessing.Tensor import run_Tensor


from TableProcessing.TableManager import (get_xyzString,
                                          add_overlapInformation,
                                          add_referenceCoordinates,
                                          add_absoluteCoordinates,
                                          unpacking_ParallelComputingResults,
                                    
                                          )   
                                          
from TableProcessing.OverlapManager import (get_centralMask, 
                                            get_overlapMask,
                                            remove_MultipleDetection
                                            )

from IO.Files.FileWriter import (save_CSV, 
                             save_Vaa3DMarker,
#                             save_Vaa3DMarker_Abs,
                             save_Figure
                             )
from IO.Image.ImageReader import (read_ImagePatch)
from IO.Image.ImageWriter import save3Dimage_as3DStack


from Plots.MultiScale import (plot_2DResultTensor, plot_InterMediateResult)
from MetaData.JSON_Manager import read_jsonDetect
import itertools

from IO.Image.ImageScanner import get_scanningCoordinates
from IO.Files.FileManager import createFolder 

from pathlib import Path

from ParallelComputing.WorkManager import parallelComputing, plot_ComputingPerformance2

# =============================================================================
# 
# =============================================================================

def run_scanDetections(pathFile):
    
    # Load Detection Settings
    args = read_jsonDetect(pathFile)
    
    #Unpack Detection Parameters
    pathFolder_ReadImage, pathFolder_WriteResults = args[0], args[1]
    scannerStart_In_px, scannerEnd_In_px = args[2], args[3]
    imgDimXYZ = args[4]
    voxelSize_In_um, voxelSize_Out_um = args[5], args[6]
    cellRadiusMin_um, cellRadiusMax_um = args[7], args[8]
    nScales = args[9]
    computingCubeSize, computingCubeOverlap = args[10], args[11]
    computeTensor = args[12]
    nProcess = args[13]
    
    # OverWrite the Saving Path to store in a SubFolder 
    rootPath = Path(pathFolder_WriteResults)
    folderName = "Detections"
    pathFolder_WriteResults   = Path.joinpath(rootPath, folderName)
    
    #Voxel Ratio: In/Out
    voxelSize_Ratio_um = voxelSize_Out_um/voxelSize_In_um
    voxelSize_Ratio_px = 1.0/voxelSize_Ratio_um
        
    # Computing Sanning Locations
    args = get_scanningCoordinates(scannerStart_In_px, scannerEnd_In_px, imgDimXYZ, voxelSize_In_um, voxelSize_Out_um, cellRadiusMax_um, computingCubeSize, computingCubeOverlap)
    scannerPositions_In_px, scannerSize_In_px, scannerOverlap_In_px = args[0], args[1], args[2]
    scannerPositions_Out_px, scannerSize_Out_px, scannerOverlap_Out_px = args[3], args[4], args[5]
    
    print()
    print('scannerPositions_In_px \n', scannerPositions_In_px)
    print('scannerPositions_Out_px \n', scannerPositions_Out_px)
    print('nPositions \n', scannerPositions_In_px.shape)
    
    # Computing Scales
    scales = get_spatialScaleRangeInPixels(cellRadiusMin_um, cellRadiusMax_um, voxelSize_Out_um, nScales)
    
    print()
    print('scales \n', scales)
    
    #??? Bypass Thresholds
    I_bk = 0
    I_bk = None
    threshold_Intensity = 0
    threshold_ScaleCount = 1
    
    #??? Debug
    debugMode = False
    # debugMode = True
    if debugMode==True:
        pathFolder_ResultsTemp = os.path.join(pathFolder_WriteResults, 'myTemp')
        createFolder(str(pathFolder_ResultsTemp), remove=True) 
    nDissectors = scannerPositions_In_px.shape[0]
    taskId = np.arange(1, nDissectors+1)
    
    myArgs = [taskId,
              list(itertools.repeat(str(pathFolder_ReadImage), nDissectors)),              
              list(itertools.repeat(str(pathFolder_WriteResults), nDissectors)),
              
              list(itertools.repeat(scales, nDissectors)),
              
              list(itertools.repeat(voxelSize_Ratio_px, nDissectors)),
              list(itertools.repeat(voxelSize_Out_um, nDissectors)),  
              
              scannerPositions_In_px,
              scannerPositions_Out_px, #remove
              
              list(itertools.repeat(scannerSize_In_px, nDissectors)),
              list(itertools.repeat(scannerSize_Out_px, nDissectors)),
              
              list(itertools.repeat(scannerOverlap_Out_px, nDissectors)),
              
              list(itertools.repeat(computeTensor, nDissectors)), 
              
              list(itertools.repeat(I_bk, nDissectors)),
              list(itertools.repeat(threshold_Intensity, nDissectors)),
              list(itertools.repeat(threshold_ScaleCount, nDissectors)),
              list(itertools.repeat(debugMode, nDissectors))
              ]   
 
    #Start the Computation          
    nProcesses, nThreads= 1, nProcess
    res = parallelComputing(func=compute_3DPoseOfCells, args=myArgs, nProcesses=nProcesses, nThreads=nThreads)
    

    #==============================================================================
    #   Unpacking the Parallel Computing Results
    #==============================================================================
    t0 = time.time()
    print()
    print('Start: Unpacking')
    start = time.time() - t0 
    df_Times, df_All, df_Central, df_Overlap, df_Origins = unpacking_ParallelComputingResults(res)
    stop = time.time() - t0 
    dt = stop-start
    print()
    print('dt:', dt)
    
    # =============================================================================
    #   removeMultipleDetection (NEW) and merge central with overlap
    # =============================================================================
    print()
    print('Start: remove_MultipleDetection')
    start = time.time() - t0 
        
    # ??? Remove Multiple Detection in Overlaping Regions  
    if nDissectors>1:    
        df_OverlapOut = remove_MultipleDetection(df_Overlap, df_Origins, scannerSize_Out_px)
    else: 
        df_OverlapOut = df_Overlap
    # df_OverlapOut = df_Overlap

    #Merge the Cell Detections of Central Regions and Overlaping Regions
    df_Detections = pd.concat([df_Central, df_OverlapOut], ignore_index=True)
    df_Detections.index += 1 
    df_Detections['ID'] = df_Detections.index
    

    stop = time.time() - t0 
    dt = stop-start
    print()
    print('dt:', dt)
      
    #==============================================================================
    #   Get the Final Results
    #==============================================================================
    print()
    print('Start: saveResults')
    start = time.time() - t0 
    
    #Save: Computing Performance (time that takes the tasks)
    fileName = '0_Origins'
    save_CSV(df_Origins, pathFolder_WriteResults, fileName) 
    
    #Save: Computing Results (Detected Cells )
    fileName = '1a_MultiScale_Detections_All'
    save_CSV(df_All, pathFolder_WriteResults, fileName)
    
    #Save: Computing Results (Detected Cells )
    fileName = '1b_MultiScale_Detections_Max'
    save_CSV(df_Detections, pathFolder_WriteResults, fileName, index=False)

    #Save: Computing Results (Detected Cells )
    fileName = '1c_MultiScale_Detections_Filter'
    save_CSV(df_Detections, pathFolder_WriteResults, fileName, index=False)

    #Save: Computing Performance (time that takes the tasks)
    fileName = '3_Times'
    save_CSV(df_Times, pathFolder_WriteResults, fileName) 

    
    stop = time.time() - t0 
    dt = stop-start
    print()
    print('dt:', dt)
    
    # =============================================================================
    # Plot Performance
    # =============================================================================
    print()
    print('Start: Plot Performance')
    start = time.time() - t0 
    
    plot_detectionPerformance(pathFolder_WriteResults, nProcesses, nThreads, scannerSize_In_px)
    
    stop = time.time() - t0 
    dt = stop-start
    print()
    print('dt:', dt)    
# =============================================================================
#     
# =============================================================================

def plot_detectionPerformance(pathFolder_WriteResults, nProcesses, nThreads, scannerSize_In_px):
       
    # Read a CSV        
    rootPath = Path(pathFolder_WriteResults)
    fileName   = "3_Times.csv" 
    pathFile   = Path.joinpath(rootPath, fileName)          
    df_Times = pd.read_csv (str(pathFile))
    
    
    computation_labels = [ '1_Read', '2_Resample',
                            '3a_D2Img','3b_MaxSpace','3c_MaxScale','3d_CountScale',
                            '4_RemoveFalse',    
                            '5a_Tensor', '5b_TensorMetrics',                                
                            '6_SplitInfo', 
                            '7_Debug'
                            ]    
    # pathFolder = os.path.join(localPath, rootName, 'Performance_CPU_IO')
    # pathFolder = pathFolder_WriteResults
    
    
    #???? Alternative
    # sortedBy = 'default'
    # fig, ax  = plot_ComputingPerformance(df_Times, computation_labels)
    sortedBy = 'taskID'
    # sortedBy = 'workerID'
    fig, ax  = plot_ComputingPerformance2(df_Times.copy(), computation_labels=computation_labels, sortedBy=sortedBy, skipStart=False)

    
    #Figure Labels
    myFontSize = 20
    figure_title = ('nWorkers='+ str(nProcesses*nThreads) + ', nProcesses='+ str(nProcesses) + ', nThreads='+ str(nThreads))
    ax.set_title(figure_title, fontsize=myFontSize)  
  
    myFontSize = 15
    ax.tick_params(axis='both', which='major', labelsize=myFontSize)
    ax.tick_params(axis='both', which='minor', labelsize=myFontSize)
    
    plt.show()
    
    #Save Figure
    pathFolder = Path(pathFolder_WriteResults)
    fileName = ('7_' + 
                str(scannerSize_In_px[0]) + 'x' + str(scannerSize_In_px[1]) + 'x' + str(scannerSize_In_px[2]) +
                '_nW_' + str(nProcesses*nThreads) + 
                '_nP_' + str(nProcesses) +
                '_nT_' + str(nThreads) 
                + '_' + sortedBy)    
    save_Figure(fig, pathFolder, fileName)                                             

# =============================================================================
#     
# =============================================================================

#def compute_3DPoseOfCells(taskId, scannerStart_In_px, scannerPosition_iso, pathFolder_WriteResults, t0, debug=False):   
def compute_3DPoseOfCells(taskId, *args, verbose=True):   
    if verbose==True:
        print()
        print("Start: compute_3DPoseOfCells()")
   
    #Unpaking the Arguments required by the Function
    pathFolder_ReadImage, pathFolder_WriteResults = args[0], args[1]
    scales = args[2]
    
    voxelSize_Ratio_px, voxelSize_Out_um = args[3], args[4]
    scannerStart_In_px, scannerStart_Out_px = args[5], args[6]    
    scannerSize_In_px, scannerSize_Out_px = args[7], args[8]
    
    scannerOverlap_Out_px = args[9]
    
    computeTensor = args[10]
    
    I_bk = args[11]    
    threshold_Intensity, threshold_ScaleCount = args[12], args[13]
    debugMode, t0 = args[14], args[15]
    
    
    #Adittional Parameters
    process_ID = os.getpid()
    thread_ID = threading.current_thread().ident

    #Addition Prameters    
    pathFolder_ResultsTemp = os.path.join(pathFolder_WriteResults, 'myTemp')


    print()
    print('----------------------')
    print('taskId:', taskId)
    print('----------------------')
    
    #==============================================================================
    # Start the Computational Routine
    #==============================================================================
    #'1_Read', '2_Resample'
    #'3a_D2Img','3b_MaxSpace','3c_MaxScale','3d_CountScale'
    #'4_RemoveFalse'    
    #'5a_Tensor', '5b_TensorMetrics'                                
    #'6_SplitInfo' 
    #'7_Debug'
    #-------------------------------------  
    #--------Get a Dissected Image--------  
    #------------------------------------- 

#    p = psutil.Process(os.getpid())
#    M = p.memory_info()[0] # in bytes 
#    print('')
#    print('Memory Usage: compute_3DPoseOfCells')
#    print (1.0*M/10**9)  
#    print(p.memory_info())

    
    # =============================================================================
    #    #Read Image Patch 
    # =============================================================================
    # #Read Image Patch
    # [imgIn, bitDepth, start, stop] = read_ImagePatch(imgPath=pathFolder_ReadImage,
    #                                       coordinates=scannerPosition_ani,
    #                                       dissectionSize=scannerSize_In,
    #                                       nThreads=1,
    #                                       showPlot=False,
    #                                       t0=t0
    #                                       )
    # if imgIn.size!=0:
    #     print('image empty')
    # scannerSize_In  = np.asarray(imgIn.shape)                            
    # compId = 1
    # op1 = [taskId, compId, process_ID, thread_ID, start, stop] 
    

    #Read Image Patch
    [imgIn, coordinates, dissectionSize, bitDepth, start, stop] = read_ImagePatch(imgPath = pathFolder_ReadImage,
                                                      coordinates = scannerStart_In_px,
                                                      dissectionSize = scannerSize_In_px,
                                                      nThreads = 1, 
                                                      showPlot=False,
                                                      t0=t0
                                                    )
    # Recompute: scannerStart_In_px, scannerStart_Out_px
    scannerStart_In_px = coordinates
    scannerStart_Out_px = (np.round(voxelSize_Ratio_px*scannerStart_In_px)).astype('int')
    
    # Recompute: InputSize, OutputSize
    scannerSize_In_px = dissectionSize
    scannerSize_Out_px = (np.round(voxelSize_Ratio_px*scannerSize_In_px)).astype('int') 
    
    
    # print()
    # print('imgIn \n', imgIn)
    # imgIn = imgIn.astype(np.float) #????
    
    # print()
    # print('imgIn \n', imgIn)
    
    # #?????? Padding if Only one Stack
    # if imgIn.shape[2]==1:
    #     scaleMax = 4*(np.ceil(scales.max())).astype(int)
    #     imgIn = np.pad(imgIn, scaleMax, 'constant')
    # scaleMax = 4*(np.ceil(scales.max())).astype(int)
    # imgIn = np.pad(imgIn, scaleMax, 'constant')
    
    # imgIn = np.pad(imgIn, 20, 'constant')
    # M = np.ones((5,5,5))
    # np.pad(M, 20, 'constant')
        
    # =============================================================================
    #     
    # =============================================================================
    
    #Check: isImageBoundary   
    # [ny, nx, nz] = imgIn.shape
    # imgDim = np.asarray([nx, ny, nz])    
    # isSizeEqual = all(imgDim==scannerSize_In_px) 
    # if not isSizeEqual:
    #     if verbose==True:
    #         print()
    #         print('The dissected Volume is on a Boundary of the Image')
        
        
    #     scannerOverlap_In_px = (np.round((1/voxelSize_Ratio_px)*scannerOverlap_Out_px)).astype('int')  
    #     d_In  = scannerStart_In_px - scannerSize_In_px/2
    #     d_In = d_In + scannerOverlap_In_px
    #     d_In[d_In<0] = 0
    #     d_Out = (np.round(voxelSize_Ratio_px*d_In)).astype('int')
        
    #     # dxyz_In = scannerSize_In_px - imgDim
    #     # dxyz_Out = (np.round(voxelSize_Ratio_px*dxyz_In)).astype('int')
        
    #     print()
    #     print('scannerStart_In_px',scannerStart_In_px)
    #     print('scannerSize_In_px', scannerSize_In_px)
    #     print('imgDimIn',  imgDim)
    #     print('d_In',  d_In)
        
    #     print()
    #     print('scannerStart_Out_px',scannerStart_Out_px)
    #     print('scannerSize_Out_px',scannerSize_Out_px)        
    #     print('imgDimOut', imgDim*voxelSize_Ratio_px)
    #     print('d_Out',  d_Out)

        
    #     #Recompute: Start (op1)
    #     # imgDim_Diff = scannerSize_In_px - imgDim
    #     # aux = scannerSize_In_px - scannerStart_In_px
    #     # aux = aux/np.abs(aux)
    #     # imgDim_Diff = aux*imgDim_Diff 
    #     # scannerStart_In_px = (np.round(scannerStart_In_px + imgDim_Diff/2)).astype('int') 
        
    #     #Recompute: Start (op2)
    #     # scannerStart_In_px = (np.round(scannerStart_In_px + imgDim/2)).astype('int') 

    #     #Recompute: Start (op3)
    #     # scannerStart_In_px = (np.round(scannerStart_In_px + imgDim/2 - d_In)).astype('int') 

    #     #Recompute: Start (op4)
    #     # print()
    #     # print('op4')
    #     # print(scannerStart_In_px - imgDim/2)
    #     # d = imgDim/2 - scannerStart_In_px  
    #     # d[d<0] = 0
    #     # scannerStart_In_px = (np.round(d)).astype('int') 
                
    #     #Recompute: Start (op5)
    #     # d = scannerStart_In_px + 0.5*(scannerSize_In_px - imgDim)
    #     # d = scannerStart_In_px + 0.5*(imgDim)
    #     # scannerStart_In_px = (np.round(d)).astype('int') 
 
    #     # Recompute: scannerStart_Out_px
    #     scannerStart_Out_px = (np.round(voxelSize_Ratio_px*scannerStart_In_px)).astype('int')
        
    #     # Recompute: InputSize
    #     scannerSize_In_px = imgDim
        
    #     # Recompute: OutputSize
    #     scannerSize_Out_px = (np.round(voxelSize_Ratio_px*scannerSize_In_px)).astype('int')
    #     # =============================================================================
    #     #         
    #     # =============================================================================

    #     print()
    #     print('--------------------')

    #     print()
    #     print('scannerStart_In_px',scannerStart_In_px)
    #     print('scannerSize_In_px', scannerSize_In_px)
    #     print('imgDimIn',  imgDim)
        
    #     print()
    #     print('scannerStart_Out_px',scannerStart_Out_px)
    #     print('scannerSize_Out_px',scannerSize_Out_px)        
    #     print('imgDimOut', imgDim*voxelSize_Ratio_px)
        
    #     #Get the Corner (upper,left,front corner) as the Origin of the Referece System    
    #     origin_Out_px = (np.round(scannerStart_Out_px - (scannerSize_Out_px - 1)/2)).astype('int')
    #     # (np.round(scannerStart_In_px - (scannerSize_In_px - 1)/2)).astype('int')
    #     origin_In_px  = (np.round(origin_Out_px/voxelSize_Ratio_px)).astype('int')
        
    #     print()
    #     print('origin_In_px', origin_In_px)
    #     print('origin_Out_px', origin_Out_px)
        
        
    # else:
    #     if verbose==True:
    #         print()
    #         print('The dissected Volume is NOT on a Boundary of the Image')
    
    
    compId = 1
    op1 = [taskId, compId, process_ID, thread_ID, start, stop] 
    
    
    # =============================================================================
    #  Forcing Isotropy
    # =============================================================================
    #Forcing the 3D-Image to be Isotropic through a Resampling Operator
    # # [imgIn, start, stop] = resample_Zdim(imgIn, z_thick, scannerSize_Out, scannerOverlap_iso, t0=t0)
    # [imgIn, start, stop]  = resample_3DImage(imgIn, scannerSize_Out, t0=t0)
    # compId = 2
    # op2 = [taskId, compId, process_ID, thread_ID, start, stop]    
    
    # #Update image dimensions after forcing Isotropy
    # [ny, nx, nz] = imgIn.shape
    # imgDim_iso = np.asarray([nx, ny, nz])  
    

    # print()
    # print('imgIn.shape', imgIn.shape)
    # print('imgIn (Min, Max)', imgIn.min(), imgIn.max())
    # print('scannerSize_Out_px', scannerSize_Out_px)    
    
    #Force Isotropy
    [imgIn, start, stop]  = resample_3DImage(imgIn, scannerSize_Out_px, t0=t0)
    
    # print()
    # print('imgIn.shape', imgIn.shape)
    # print('imgIn (Min, Max)', imgIn.min(), imgIn.max())
    # sys.exit()
    
    compId = 2
    op2 = [taskId, compId, process_ID, thread_ID, start, stop]    
    
    # #Update image dimensions after forcing Isotropy
    # [ny, nx, nz] = imgIn.shape
    # imgDim_Out_px = np.asarray([nx, ny, nz])  
    
    
    #------------------------------------- ------- 
    #--------Run the Detection Algorithm----------  
    #---------------------------------------------
    
    #1) Cell Location Algorithm: xyz-coordinates of the geometric center of the soma
    rS = 1.10
    # rS = 1.6    #suitable for detecting closed cells (worst)
    # rS = 2.0    #suitable for detecting closed cells (optimal)
    # rS = 6.0    #suitable for detecting closed cells (worst)
    # rS = 9.0    #suitable for detecting closed cells (worst)
    rV = 1.0
    imgDoGMS, df_All, df_Cells0, op = run_CellLocalizer(imgIn, scales, rS, rV, I_bk, t0=t0)
    op3a, op3b, op3c, op3d = op
    op3a = [taskId, 3, process_ID, thread_ID, op3a[0], op3a[1]] 
    op3b = [taskId, 4, process_ID, thread_ID, op3b[0], op3b[1]]  
    op3c = [taskId, 5, process_ID, thread_ID, op3c[0], op3c[1]] 
    op3d = [taskId, 6, process_ID, thread_ID, op3d[0], op3d[1]] 
    
    # df_All['Gain'] = (df_All['I_DoG'] - df_All['I_Pro'])
    # df_Cells0['Gain'] = (df_Cells0['I_DoG'] - df_Cells0['I_Pro'])  
    df_Cells0['Gain'] = (df_Cells0['I_DoG'] - df_Cells0['I_Pro'])/(df_Cells0['I_DoG'] + df_Cells0['I_Pro'])
    # df_Detections['R_um'] = df_Detections['S']*voxelSize_um_Out[0]
    
    
    
    #2) Remove: False Positive Cells
    #2.a) Remove detected cells with Low Local Intensities 
    #Note: the threshold is applied over I_DoG (not I0)
    [df_Cells1, start, stop] = remove_FalseCells_with_LowIntensity(df_Cells0, threshold_Intensity, mode='absolute', t0=t0)
    op4 = [taskId, 7, process_ID, thread_ID, start, stop] 
    
    #2.b) Remove detected cells with Low Scale Count
    [df_Cells2, start, stop] = remove_FalseCells_with_LowScaleCount(df_Cells1, scaleCountThreshold=threshold_ScaleCount)

    #???? Bypassing the threshold of n_scales ()
    # df_Cells2 = df_Cells1.copy()
    # df_Cells2 = df_Cells0.copy()


    # print()
    # print("cells detected")
    # print(df_Cells2)
    
    #------------------------------------- ------- 
    #--------Run the Tensor Algorithm----------  
    #---------------------------------------------     
    op5a = np.zeros(6)
    op5b = np.zeros(6)
    if computeTensor==True:
        #3) Compute Tensor Metrics: Orientation, Anisotropy, Tubularity, Disck
        [df_Cells3, op] = run_Tensor(imgIn, imgDoGMS, df_Cells2, scales, t0=t0)
        op5a, op5b = op
        op5a = [taskId, 8,         process_ID, thread_ID, op5a[0], op5a[1]] 
        op5b = [taskId, 9,  process_ID, thread_ID, op5b[0], op5b[1]] 
        
        #4a) Selecting the Results to Store
        df_Cells = df_Cells3
        
    else: 
        op5a = [taskId, 8,         process_ID, thread_ID, op5a[0], op5a[1]] 
        op5b = [taskId, 9,  process_ID, thread_ID, op5b[0], op5b[1]] 
        
        #4b) Selecting the Results to Store
        df_Cells = df_Cells2

    
#    p = psutil.Process(os.getpid())
#    M = p.memory_info()[0] # in bytes 
#    print('')
#    print('Memory Usage: After Tensor')
#    print (1.0*M/10**9)  
#    print(p.memory_info())


# =============================================================================
#     Compute Absolute Origin 
# =============================================================================
    start = time.time()-t0
    # #Relative
    # x_rel_out_px
    # x_rel_in_px
    # x_rel_um
    
    # #Absolute
    # X_abs_out_px
    # X_abs_in_px
    # X_abs_um
    
    #Get 
    scannerStart_Out_px = (np.round(voxelSize_Ratio_px*scannerStart_In_px)).astype('int')
    
    #Get the Corner (upper,left,front corner) as the Origin of the Referece System    
    origin_Out_px = (np.round(scannerStart_Out_px - (scannerSize_Out_px - 1)/2)).astype('int')
    # (np.round(scannerStart_In_px - (scannerSize_In_px - 1)/2)).astype('int')
    origin_In_px  = (np.round(origin_Out_px/voxelSize_Ratio_px)).astype('int')
    
    print()
    print('scannerStart_In_px', scannerStart_In_px)
    print('scannerStart_Out_px', scannerStart_Out_px)
    
    print()
    print('origin_In_px', origin_In_px)
    print('origin_Out_px', origin_Out_px)
    
    df_Cells['X_in_px'] =  (np.round(df_Cells['X']/voxelSize_Ratio_px[0])).astype('int')
    df_Cells['Y_in_px'] =  (np.round(df_Cells['Y']/voxelSize_Ratio_px[1])).astype('int')
    df_Cells['Z_in_px'] =  (np.round(df_Cells['Z']/voxelSize_Ratio_px[2])).astype('int')
    
    df_Cells['X_um'] =  df_Cells['X']*voxelSize_Out_um[0]
    df_Cells['Y_um'] =  df_Cells['Y']*voxelSize_Out_um[1]
    df_Cells['Z_um'] =  df_Cells['Z']*voxelSize_Out_um[2]

    df_Cells['X_abs_out_px'] = (origin_Out_px[0] + df_Cells['X']) #IMP
    df_Cells['Y_abs_out_px'] = (origin_Out_px[1] + df_Cells['Y'])
    df_Cells['Z_abs_out_px'] = (origin_Out_px[2] + df_Cells['Z']) 
    
    df_Cells['X_abs_in_px'] = (origin_In_px[0] + df_Cells['X_in_px']) #IMP    
    df_Cells['Y_abs_in_px'] = (origin_In_px[1] + df_Cells['Y_in_px'])
    df_Cells['Z_abs_in_px'] = (origin_In_px[2] + df_Cells['Z_in_px']) 
    
    df_Cells['X_abs_um'] = df_Cells['X_abs_out_px']*voxelSize_Out_um[0]
    df_Cells['Y_abs_um'] = df_Cells['Y_abs_out_px']*voxelSize_Out_um[1]
    df_Cells['Z_abs_um'] = df_Cells['Z_abs_out_px']*voxelSize_Out_um[2]
    
    #Size
    df_Cells['S_um'] = df_Cells['S']*voxelSize_Out_um[0]
    
  
    
    
    
    # #Size
    # df_Cells['S_um'] = df_Cells['S']*voxelSize_Out_um[0]
    # origin_Out_px
    
    # Vaa3D
    # ??? mask
    # mask = (df_Cells['N']>=9)
    # df_Cells = df_Cells[mask]
    df_Vaa3D = pd.DataFrame()
    
    #When save it as a tiff
    # df['X'] = df_Cells['Z'] + 1
    # df['Y'] = df_Cells['X'] + 1
    # df['Z'] = df_Cells['Y'] + 1
    
    df_Vaa3D['X'] = df_Cells['X_abs_in_px'] + 1
    df_Vaa3D['Y'] = df_Cells['Y_abs_in_px'] + 1
    df_Vaa3D['Z'] = df_Cells['Z_abs_in_px'] + 1
    
    # 'Z', 'X', 'Y']
    #Other Features
    n = df_Cells.shape[0]
    df_Vaa3D['R'] = 1*df_Cells['S']     
    df_Vaa3D['shape'] = np.zeros(n)
    df_Vaa3D['name'] =  df_Cells.index.values
    df_Vaa3D['comment'] = n*['0']
    df_Vaa3D['cR'] = 255*np.ones(n)
    df_Vaa3D['cG'] = 0*np.ones(n)
    df_Vaa3D['cB'] = 0*np.ones(n)
    
    df_Vaa3D = df_Vaa3D.astype(int)

    # =============================================================================
    #     All
    # =============================================================================
    df_All['X_abs_out_px'] = (origin_Out_px[0] + df_All['X']) #IMP
    df_All['Y_abs_out_px'] = (origin_Out_px[1] + df_All['Y'])
    df_All['Z_abs_out_px'] = (origin_Out_px[2] + df_All['Z'])   
    
    df_All['X_ref_out_px'] = origin_Out_px[0]  
    df_All['Y_ref_out_px'] = origin_Out_px[1] 
    df_All['Z_ref_out_px'] = origin_Out_px[2] 

    #==============================================================================
    # Extract Detections from the Central Region separetely from the Overalp Region
    #==============================================================================
    #Add the Origin Coordinate to the current Volume
    # df_Cells = add_referenceCoordinates(df_Cells, origin=origin_Out_px)
    df_Cells['X_ref_in_px'] = origin_In_px[0]  
    df_Cells['Y_ref_in_px'] = origin_In_px[1] 
    df_Cells['Z_ref_in_px'] = origin_In_px[2] 
    
    df_Cells['X_ref_out_px'] = origin_Out_px[0]  
    df_Cells['Y_ref_out_px'] = origin_Out_px[1] 
    df_Cells['Z_ref_out_px'] = origin_Out_px[2] 




    #Get all Detection Coordinates
    xyzCells = df_Cells['X'].values, df_Cells['Y'].values, df_Cells['Z'].values 
    
    #Extract Detections from the Central Region (i.e. Non-Overlaping Region)  
    maskCenter = get_centralMask(xyzCells, origin_Out_px, scannerOverlap_Out_px, scannerSize_Out_px)        
    df_central = df_Cells[maskCenter]
    
    #Extract Detections from the Overlap Region 
    maskOverlap = get_overlapMask(xyzCells, origin_Out_px, scannerOverlap_Out_px, scannerSize_Out_px)
    df_overlap = df_Cells[maskOverlap]

    stop = time.time()-t0
    op6 = [taskId, 10, process_ID, thread_ID, start, stop] 
    
    
# =============================================================================
# 
# =============================================================================
#     #==============================================================================
#     #   Add Coordinates Information to the Table (i.e. df_cells) 
#     #==============================================================================
#     start = time.time() - t0
#     #Get the Corner (upper,left,front corner) as the Origin of the Referece System
#     xyz_corner_iso = scannerPosition_iso - (imgDim_iso - 1)/2 

#     print('scannerPosition_iso=', scannerPosition_iso)
#     print('xyz_corner_iso=', xyz_corner_iso)

#     #==============================================================================
#     # 
#     #==============================================================================
#     #Add the Origin Coordinate to the current Volume
#     df_Cells = add_referenceCoordinates(df_Cells, origin=xyz_corner_iso)
    
#     #Add the Absolute Coordinates of the Detections to the current Volume
#     df_Cells = add_absoluteCoordinates(df_Cells, origin=xyz_corner_iso, dissection_Ratio=dissection_Ratio)         
        
    
#     #==============================================================================
#     # Extract Detections from the Central Region separetely from the Overalp Region
#     #==============================================================================
#     #Get all Detection Coordinates
#     xyzCells = df_Cells['X'].values, df_Cells['Y'].values, df_Cells['Z'].values 
    
#     #Extract Detections from the Central Region (i.e. Non-Overlaping Region)  
#     maskCenter = get_centralMask(xyzCells, xyz_corner_iso, scannerOverlap_iso, imgDim_iso)        
#     df_central = df_Cells[maskCenter]
    
#     #Extract Detections from the Overlap Region 
#     maskOverlap = get_overlapMask(xyzCells, xyz_corner_iso, scannerOverlap_iso, imgDim_iso)
#     df_overlap = df_Cells[maskOverlap]

#     #Add Overlap Information to the Overlap Table
# #    df_overlap = add_overlapInformation(df_overlap, xyz_corner_iso, scannerOverlap_iso, imgDim_iso)

#     stop = time.time()-t0
#     op6 = [taskId, 10, process_ID, thread_ID, start, stop] 
    
    
    #==============================================================================
    #    Save Debuugin Results   
    #==============================================================================
    start = time.time()-t0    
    if debugMode==True:
        #Set the NameID to save the files
        fileName = get_xyzString(origin_Out_px)
        
        #Save Table: Central Region + Overlap Region
        sub_fileName = get_xyzString(origin_Out_px) + '_BothRegion'
        save_CSV(df_Cells, pathFolder_ResultsTemp, sub_fileName)        
    
        #Save Table: Central Region
        sub_fileName = get_xyzString(origin_Out_px) + '_CentralRegion'
        save_CSV(df_central, pathFolder_ResultsTemp, sub_fileName)
        
        #Save Table: Overlap Region      
        sub_fileName = get_xyzString(origin_Out_px) + '_OverlapRegion'
        save_CSV(df_overlap, pathFolder_ResultsTemp, sub_fileName)        
        
        
        #Save 3D-Image: 3D-tiff 
        imgIn[imgIn<0]=0
        if bitDepth==8:
            imgIn_uint = imgIn.astype(np.uint8)
        elif bitDepth==16:
            imgIn_uint = imgIn.astype(np.uint16)
        elif bitDepth==32:
            imgIn_uint = imgIn.astype(np.uint32)
        else:
            print('Unknown bitDepth image format')   
        
        imgIn_uint = imgIn_uint.transpose([2, 0, 1])
        save3Dimage_as3DStack(imgIn_uint, pathFolder_ResultsTemp, fileName)        
        
        #Save Vaa3D Marker File 
        # save_Vaa3DMarker(df_Cells, imgDim_iso, pathFolder_ResultsTemp, fileName) 
            
        # #Saving the Marker File
        # folderPath = pathFolder_ResultsTemp
        # fileName = 'vaa3D'
        # fileExtension = '.marker'       
        # filePath = os.path.join(folderPath, fileName + fileExtension)  
        # df_Vaa3D.to_csv(filePath, sep=',', encoding='utf-8', index=False, header=True)     
        
        # #Plot Detected Cells
        # fig, axs = plot_2DResultTensor(imgIn, df_Cells, scannerOverlap_Out_px)
        # fig.tight_layout(h_pad=1.0) 
        # fig_title = fileName + ', N_cells=' + str(df_Cells.shape[0])
        # axs[1].set_title(fig_title, fontsize=16)
        # plt.show()     
        
        # #Save the Plot
        # save_Figure(fig, pathFolder_ResultsTemp, fileName)
    
        #Visualizing Debugging Plots (InterMediate Results for Debugging)
        # plot_InterMediateResult(imgIn, df_All, df_Cells0, df_Cells1, df_Cells2, df_Cells3, scannerOverlap_Out_px)


        #Save OutPut Image for Debuging    
        for i in range(0, imgDoGMS.shape[0]):        
            #Save 3D-Image 
            imgSave = imgDoGMS[i].copy()
            imgSave[imgSave<0]=0
            imgSave = (np.round(imgSave)).astype(np.uint16)
            
            #Transpose for Saving ????
            imgSave = imgSave.transpose([2, 0, 1])
            
            fileName = get_xyzString(origin_Out_px) + '_ImageOut_' + str(scales[i])
            save3Dimage_as3DStack(imgSave, pathFolder_ResultsTemp, fileName)
        
        
    stop = time.time()-t0
    op7 = [taskId, 11, process_ID, thread_ID, start, stop] 
        
    #==============================================================================
    #     Times
    #==============================================================================
    M_times = [op1, op2, op3a, op3b, op3c, op3d, op4, op5a, op5b, op6, op7]
    df_Times = pd.DataFrame(M_times, columns=['taskID', 'compID', 'processID', 'threadID', 'start', 'stop'])
    df_Times['width'] = df_Times['stop'] - df_Times['start']
    
    # =============================================================================
    #     Origins
    # =============================================================================
    # M_Origins = [[origin_In_px[0],  origin_In_px[1],  origin_In_px[2],
    #               origin_Out_px[0], origin_Out_px[1], origin_Out_px[2]]]
    # cols = ['X_ref_in_px',  'Y_ref_in_px',  'Z_ref_in_px',
    #         'X_ref_out_px', 'Y_ref_out_px', 'Z_ref_out_px']
    # df_Origins = pd.DataFrame(M_Origins, columns=cols)
    M_Origins = [[origin_In_px[0],  origin_In_px[1],  origin_In_px[2],
                  scannerSize_In_px[0], scannerSize_In_px[1], scannerSize_In_px[2],
                  origin_Out_px[0], origin_Out_px[1], origin_Out_px[2],
                  scannerSize_Out_px[0], scannerSize_Out_px[1], scannerSize_Out_px[2]
                  ]]
    cols = ['X_ref_in_px',  'Y_ref_in_px',  'Z_ref_in_px',
            'X_ScannerSize_In_px',  'Y_ScannerSize_In_px',  'Z_ScannerSize_In_px',
            'X_ref_out_px', 'Y_ref_out_px', 'Z_ref_out_px',
            'X_ScannerSize_Out_px',  'Y_ScannerSize_Out_px',  'Z_ScannerSize_Out_px']
    df_Origins = pd.DataFrame(M_Origins, columns=cols)


    # =============================================================================
    #     Return
    # =============================================================================

    if verbose==True:
        print()
        print("Stop: compute_3DPoseOfCells()")
        
    return df_Times, df_All,  df_central, df_overlap, df_Origins


if __name__== '__main__':
    pass

#    #RAM
#    pid = os.getpid()
#    py = psutil.Process(pid)
#    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
#    print('memory use:', memoryUse) 





#    print('')
#    print('imgIn_Dim=', imgIn.shape) 
##    jajaja
    
   
    # #Intenstity Mapping: [0...128...255] -> [-1...0...+1]    
    # imgIn = change_ImageIntensityMap(imgIn, x0=0, x1=2**bitDepth-1, y0=-1, y1=+1)
    # imgIn = imgIn - imgIn.mean()
    # imgIn = imgIn.astype(np.float32)
    
    # print()
    # print("I_Min, I_Max")
    # print(imgIn.min(), imgIn.max())
    # print("I_Type")
    # print(type(imgIn[0,0,0]))
    
   
#    p = psutil.Process(os.getpid())
#    M = p.memory_info()[0] # in bytes 
#    print('')
#    print('Memory Usage: compute_3DPoseOfCells: read input')
#    print (1.0*M/10**9)  
#    print(p.memory_info())

    #Center the Dynamic Range (subtracting the mean luminance)    
    #The z score tells you how many standard deviations from the mean your score is
#    imgIn = imgIn -  imgIn.mean()  
    