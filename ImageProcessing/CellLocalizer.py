# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:30:28 2020

@author: pc
"""
import numpy as np
import pandas as pd
import time


#from ImageFilters import get_DoG
from ImageProcessing.ImageFilters import get_DoG, get_Gaussian
from skimage.morphology import opening, closing, erosion, dilation
from skimage.filters.rank import maximum, minimum
from skimage.feature import peak_local_max
from ImageProcessing.IntensityMapping import change_ImageIntensityMap


from scipy import signal        
from scipy import ndimage


import sys

#import cv2

def sphere(n):
    n = np.ceil(n).astype(int)
    # print()
    # print('Sphere N', n)
    struct = np.zeros((2 * n + 1, 2 * n + 1, 2 * n + 1))
    x, y, z = np.indices((2 * n + 1, 2 * n + 1, 2 * n + 1))
    mask = (x - n)**2 + (y - n)**2 + (z - n)**2 <= n**2
    struct[mask] = 1
    struct = struct.astype(np.uint8)
    return struct
#==============================================================================
#Note: Maximum (singular); Maxima (plural)
#Note: Scale is given as the Radius of the Sphere that encolses the 3D-Object
#==============================================================================

#==============================================================================
# Main Routine
#==============================================================================

def run_CellLocalizer(imgIn_Raw, scales, rS, rV, I_bk=None, t0=0):
    
    #Cast Variables
    scales = np.asarray(scales)    
    
    # #1.0) Compute the Second Derivative of the Input Image at several Spatial Scales
    imgIn_Pro = compute_PreProcessing(imgIn_Raw, I_bk)
    # op0 = [start, stop]
    
    #1.1) Compute the Second Derivative of the Input Image at several Spatial Scales
    imgDoGMS, start, stop = compute_SecondDerivativeMS(imgIn_Pro, scales, rS, rV, t0=t0)
    op1 = [start, stop]
   
    #1.2) Compute the Local Maxima at each Spatial Scale 
    df_All, start, stop = compute_SpatialMaximaMS(imgIn_Raw, imgIn_Pro, imgDoGMS, scales, I_threshold=0.0, t0=t0) 
    op2 = [start, stop]                                                         
    
    #1.3) Compute the Maximum along Spatial Scales
    df_Cells, start, stop = compute_ScaleMaximumMS(df_All, scales, t0=t0)
    op3 = [start, stop] 
    
    #1.4) Compute the Number of times that the same Cell is detected at several Spatial Scales
    df_Cells, start, stop = compute_ScaleCount(df_All, df_Cells, t0=t0)
    op4 = [start, stop] 
    
    #Computing Times
    op = [op1, op2, op3, op4]
    return imgDoGMS, df_All, df_Cells, op


#==============================================================================
#   Subrutines
#==============================================================================
def compute_PreProcessing(imgIn, I_bk=None):
    
    # #Intenstity Mapping: [0...128...255] -> [-1...0...+1]  
    # imgIn = imgIn.astype(np.float64)
    # bitDepth = 16
    # imgIn = change_ImageIntensityMap(imgIn, x0=0, x1=2**bitDepth-1, y0=-1.0, y1=+1.0)
    # imgIn = change_ImageIntensityMap(imgIn, x0=imgIn.min(), x1=imgIn.max(), y0=-1.0, y1=+1.0)
    
    # Gaussian
    # kg = 1.0 
    # s = 3
    # s = np.asarray([s, s, s])      
    # s_gauss = kg*np.asarray(s)
    # Fxyz = get_Gaussian(s_gauss, a=1.0) 
    # imgIn = signal.convolve(imgIn, Fxyz, "same")
    
    # # Center the Dynamic Range (around I_mean)
    # Note: This is importat to remove the "Boudarie Effects"

    #If not background is provided... estimated as the mean value
    if I_bk==None:
        I_bk = imgIn.mean()  + 0.5*imgIn.std()
        # print()
        # print('Automatic I_bk=', I_bk)
    
     #Automatic Determination of the Background  
    # I_bk = imgIn.mean()  + 0.5*imgIn.std()
    
    # # ??? Center the Dynamic Range
    imgOut = imgIn - I_bk
    
    
    # ???? Remove Background (Thresholding)
    imgOut[imgOut<=0] = -1
    # imgOut[imgOut<=0] = 0
    
    # # Gaussian
    # kg = 1.0 
    # s = 2
    # s = np.asarray([s, s, s])      
    # s_gauss = kg*np.asarray(s)
    # Fxyz = get_Gaussian(s_gauss, a=1.0) 
    # imgIn = signal.convolve(imgIn, Fxyz, "same")
        
    # #Remove Background
    # imgIn[imgIn<=I_bk] = I_bk 
    # imgIn[imgIn<=I_bk] = 0
    
    # # # ??? Center the Dynamic Range
    # imgIn = imgIn - I_bk    
    # # ??? Thresholding
    # imgIn[imgIn<=0] = - I_bk
    
    # ??? Binarization
    # imgIn = imgIn - I_bk
    # imgIn[imgIn<=0] = -1
    # imgIn[imgIn>0] = 1
 
    # # Verbose
    # print()
    # print('compute_PreProcessing')
    # print('I_mean=',  np.mean(imgOut))
    # print('I_median=', np.median(imgOut))
    # print('I_min=', imgOut.min())
    # print('I_max=', imgOut.max())
    
    # Draft
    # change_ImageIntensityMap(np.array([[0, 1], [0,0]]), x0=0, x1=2**bitDepth-1, y0=-1.0, y1=+1.0)
    # change_ImageIntensityMap(np.array([[0, 1], [0,0]]), x0=0, x1=1, y0=-1.0, y1=+1.0)
    
    return imgOut

#1.1) Compute the Second Derivative of the Input Image at several Spatial Scales
#Note: The second derivative is computed through a DoG Filter instead of a LoG Filter 
def compute_SecondDerivativeMS(imgIn, scales, rS, rV, t0=0):
    start = time.time() - t0 
    
    #General Initialization
    ns = scales.shape[0]
    imgDoGMS = np.empty(ns, dtype=object)
   
    for i in range(0, ns): 
        #Get the i-th scale 
        scale = scales[i]
        # scale = scales[i]/1.5
        s = np.array([scale, scale, scale])
        # s_gauss = s/np.sqrt(2)
        s_DoG   = s
        
        # # # # ??? Binarization
        # imgOut = imgIn - 2000
        # imgOut[imgOut<=0] = -1
        # imgOut[imgOut>0] = 1
        
        # # # ???? Distance Transform
        # from scipy import ndimage
        # imgOut = ndimage.distance_transform_edt(imgOut)
        
        # print(img3D_Out.min(), img3D_Out.max())
        # jajaj
        
        # # ???? Opening
        # from skimage.morphology import opening
        # imgOut = opening(imgOut, sphere(n=scale/2))
        
        # # ???? Opening
        # from scipy.ndimage import grey_opening 
        # imgOut = grey_opening(imgOut, structure=sphere(n=scale/2))
        
        # # # ??? Smoothing
        # from ImageProcessing.ImageFilters import  get_Gaussian
        # from scipy import signal
        # s_Gauss = np.array([scale,scale,scale])
        # Fxyz = get_Gaussian(s=s_Gauss/2)
        # imgOut = signal.convolve(imgOut, Fxyz, "same")
        
        # =============================================================================
        #   OP1      
        # =============================================================================
        # # ???? Opening
        # from scipy.ndimage import grey_opening 
        # imgOut = grey_opening(imgIn, structure=sphere(n=scale/2))
        
        # # # ??? Binarization
        # imgOut = imgOut - 2000
        # imgOut[imgOut<=0] = -1
        # imgOut[imgOut>0] = 1        
        
        # # # ???? DoG
        # from ImageProcessing.ImageFilters import  get_DoG
        # from scipy import signal
        # s_DoG = np.array([scale,scale,scale])
        # F_DoG = get_DoG(s_DoG, rS=1.1, rV=1.0)
        # imgOut = signal.convolve(imgOut, F_DoG, "same")
    
        # =============================================================================
        #   OP2    
        # =============================================================================
        # # # ??? Center the Dynamic Range
        # imgOut = imgIn - 2000
       
        # # # ???? DoG
        # from ImageProcessing.ImageFilters import  get_DoG
        # from scipy import signal
        # s_DoG = np.array([scale,scale,scale])
        # F_DoG = get_DoG(s_DoG, rS=1.1, rV=1.0)
        # imgOut = signal.convolve(imgOut, F_DoG, "same")
        
        # # ?????? Rectification
        # imgOut[imgOut<0] = 0
        
        # =============================================================================
        #   Op3      
        # =============================================================================
        # # # ??? Center the Dynamic Range
        # I_bk = 2000
        # imgOut = imgIn - I_bk
        
        # ??? Thresholding
        # imgOut[imgOut<=0] = 0
        # imgOut[imgOut<=0] = -I_bk
        
        # # ???? Binarized
        # imgOut[imgOut<=0] = -1
        # imgOut[imgOut>0]  = +1
        
        # # ??? Smoothing
        # from ImageProcessing.ImageFilters import  get_Gaussian
        # from scipy import signal
        # s_Gauss = np.array([scale,scale,scale])
        # Fxyz = get_Gaussian(s=s_Gauss)
        # imgOut = signal.convolve(imgOut, Fxyz, "same")
        
        # # ???? DoG
        # from ImageProcessing.ImageFilters import  get_DoG
        # from scipy import signal
        # s_DoG = np.array([1.0, 1.0, 1.0])
        # F_DoG = get_DoG(s_DoG, rS=1.1, rV=1.0)
        # imgOut = signal.convolve(imgOut, F_DoG, "same")
        
        # # ???? Rectification
        # imgOut[imgOut<0] = -1
        
        # # # ??? Smoothing
        # from ImageProcessing.ImageFilters import  get_Gaussian
        # from scipy import signal
        # s_Gauss = np.array([scale,scale,scale])
        # Fxyz = get_Gaussian(s=s_Gauss/2.)
        # imgOut = signal.convolve(imgOut, Fxyz, "same")
        
        # # ???? Closing
        # from skimage.morphology import closing
        # imgOut = closing(imgOut, sphere(n=scale/2.0))
        
        # # ???? Opening
        # from skimage.morphology import opening
        # imgOut = opening(imgOut, sphere(n=scale/2.0))
 
        # =============================================================================
        #   Op4      
        # =============================================================================


        # # ??? Smoothing
        # s_Gauss = np.array([scale,scale,scale])
        # Fxyz = get_Gaussian(s=s_Gauss/2.)
        # imgOut = signal.convolve(imgIn, Fxyz, "same")
        
        # #Morphological Operations        
        # # myDisk = sphere(n=scale/1.0)        
        # # imgOut = erosion(imgOut, myDisk)
        # # imgOut = dilation(imgOut, myDisk)
        # # imgOut = erosion(imgOut, myDisk)
        
        # # ???? Closing         
        # # imgOut = closing(imgOut, myDisk)
        
        # # ???? Opening
        # myDisk = sphere(n=scale/2.0) 
        # imgOut = opening(imgOut, myDisk)
        

        # # ???? DoG
        # s_DoG = np.array([scale,scale,scale])
        # F_DoG = get_DoG(s_DoG, rS=rS, rV=rV)
        # imgOut = signal.convolve(imgOut, F_DoG, "same")
        
        # # imgOut = erosion(imgOut, myDisk)
        
        # # ???? Rectification
        # imgOut[imgOut<=0] = -1
        
 
        
        # ???? Dithering 
        # imgOut[imgOut<0]=0
        # imgOut = (np.round(imgOut)).astype(np.uint16)
        
        # imgOut[imgOut<0]=0
        # imgOut = (np.round(imgOut)).astype(np.uint16)
        # imgOut = maximum(minimum(imgOut, myDisk), myDisk) #Closign
        # imgOut = minimum(maximum(imgOut, myDisk), myDisk) #Opening
        
        # =============================================================================
        #   Op5     
        # =============================================================================
        # ??? Smoothing
        # s_Gauss = np.array([scale,scale,scale])
        # Fxyz = get_Gaussian(s=s_Gauss/2.)
        # imgOut = signal.convolve(imgIn, Fxyz, "same")
        
        # #Morphological Operations 
        # myDisk = sphere(n=scale/2.0)
        
        # # ???? Opening         
        # imgOut = opening(imgOut, myDisk)
        
        # # ???? DoG
        # s_DoG = np.array([scale,scale,scale])
        # F_DoG = get_DoG(s_DoG, rS=rS, rV=rV)
        # imgOut = signal.convolve(imgOut, F_DoG, "same")
        
        # # ???? Erosion 
        # imgOut = erosion(imgOut, myDisk)
                
        # # ???? Rectification
        # imgOut[imgOut<=0] = -1        
        
        # =============================================================================
        #   Default      
        # =============================================================================
        
        # ???? DoG
        s_DoG = np.array([scale,scale,scale])
        F_DoG = get_DoG(s_DoG, rS=rS, rV=rV)
        imgOut = signal.convolve(imgIn, F_DoG, "same")
        
        
        # ???? Rectification
        imgOut[imgOut<0] = -1
        
        # =============================================================================
        # Op: Basic
        # =============================================================================
        
        # #Get the 3D-DoG Filter at the i-th scale (Isotropic Filter)        
        # F_DoG = get_DoG(s_DoG, rS, rV)
        
        # # #Op1
        # # t0 = time.time()
        # imgOut = signal.convolve(imgIn, F_DoG, "same")
        # # imgOut = signal.convolve(imgIn, F_DoG, "valid")
        # # imgOut = signal.convolve(imgIn, F_DoG, "full") 
        # dt = time.time() - t0
        # print()
        # print('dt')
        # print(dt)
        
        # # #Op2: Takes too much time
        # # t0 = time.time()
        # # imgOut = ndimage.convolve(imgIn, F_DoG, mode='constant', cval=0.0)
        # # dt = time.time() - t0
        # # print()
        # # print('dt')
        # # print(dt)
        
        # # # ?????? Rectification 
        # imgOut[imgOut<0] = 0.000000
        
        # =============================================================================
        #         
        # =============================================================================
        
        # print('')   
        # print('scale') 
        # print(scale)
        # print('imgInDim vs filterDim vs imgOutDim') 
        # print(imgIn.shape, F_DoG.shape, imgOut.shape)
        # print('imgIn : Imin, Imax') 
        # print(imgIn.min(), imgIn.max())
        # print('ImgOut DoG: Imin, Imax') 
        # print(imgOut.min(), imgOut.max())

        # #Visualize
        # from matplotlib import pyplot as plt
        # import matplotlib.cm as cm
        # img2D = imgOut[:, :,imgIn.shape[2]//2]
        # ix = (np.where(imgOut==imgOut.max()))
        # plt.imshow(img2D, cm.Greys_r, interpolation='nearest')
        # plt.scatter(ix[1], ix[0])
        # plt.show()
        # # # # jajaj
        
        # =============================================================================
        #     Cheking Dynamic Range     ????
        # =============================================================================
        
        # #Save 3D-Image 
        # imgSave = imgOut.copy()
        # imgSave[imgSave<0]=0
        # imgSave = (np.round(imgSave)).astype(np.uint8)
        
        
        # print()
        # print('scale=', scale)
        # print('ImgIn : Imin, Imax') 
        # print(imgIn.min(), imgIn.max())
        # print('ImgOut : Imin, Imax') 
        # print(imgOut.min(), imgOut.max())
        # print('ImgSave : Imin, Imax') 
        # print(imgSave.min(), imgSave.max())
        

                
        # =============================================================================
        #         
        # =============================================================================

        #Store the Output at the i-th scale        
        imgDoGMS[i] = imgOut         
        # break

    stop = time.time() - t0 
    return imgDoGMS, start, stop

#1.2) Compute the Local Maxima at all Spatial Scales     
def compute_SpatialMaximaMS(imgIn_Raw, imgIn_Pro, imgDoGMS, scales, I_threshold, t0=0):
    start = time.time() - t0     
    
    #General Initialization
    ns = scales.shape[0]
    table = [] 
    
    for i in range(0,ns):
        #Get the Second Derivative Scale
        imgOut = imgDoGMS[i]
        scale = scales[i]
        
        #Compute: Local Maxima of hte 
        yxz_coord, I_Raw, I_Pro, I_DoG = compute_SpatialMaxima(imgIn_Raw, imgIn_Pro, imgOut, scale, I_threshold)        
        
        # print(yxz_coord)
        # sys.exit()
        
        #Store the                  
        S = scale*np.ones(I_Raw.shape[0])
        # data = np.column_stack((yxz_coord, S, I_Raw, I_Pro, I_DoG))
        yxz_coord[:,[0, 1]] = yxz_coord[:,[1, 0]] 
        data = np.column_stack((yxz_coord, S, I_Raw, I_Pro, I_DoG))
        
        if not len(table):
            table = data
        else:       
            table = np.vstack((table, data))
            
#        print('')
#        print('debug')
#        print('i', i)
#        print('scale', scale)
#        print('data', data)
#        print(table)
    
    columns = ['X', 'Y', 'Z', 'S', 'I_Raw', 'I_Pro', 'I_DoG']
    # columns = ['Y', 'X', 'Z', 'S', 'I_Raw', 'I_Pro', 'I_DoG']
    
#(df_Cells['I']/df_Cells['I0'])

    df = pd.DataFrame(table, columns=columns)
    stop = time.time() - t0 
    return df, start, stop
    
    
#1.2.1) Compute the Local Maxima at a single Spatial Scale   
def compute_SpatialMaxima(imgIn_Raw, imgIn_Pro, imgOut, scale, I_threshold=0.0, k=1.0):
    
    # ??? Get the radius of the Spheric Volume over which compute the Maximum   
    k=0.10  #best for closer neurons
    k=0.25
    # k=0.5
    k=1.0
    # k=1.5
    # k=2.0
    d_min = int(np.round(k*scale)) 
    if d_min <=1:
        d_min=1
    # print('d_min')
    # print()
    
    # I_threshold = imgOut.mean()
    # print('')
    # print('compute_SpatialMaxima')
    # print('I_out_avg', I_threshold)
    
    # print()
    # print('scale', scale)
    # print('imgIn : Imin, Imax') 
    # print(imgIn_Raw.min(), imgIn_Raw.max())
    # print('ImgOut DoG: Imin, Imax') 
    # print(imgOut.min(), imgOut.max())
    # print('imgOut shape', imgOut.shape)


    #Compute: Local Maxima at  the i-th scale (peak_idx -> xyz_coord)   
    peak_idx = peak_local_max(imgOut,
                              min_distance=int(np.round(d_min)),
                              threshold_abs=I_threshold,
                               # exclude_border=int(np.round(1*scale)) # ???
                              )  
    
    # if imgOut.shape[2]==1:
        
    #     #Compute: Local Maxima at  the i-th scale (peak_idx -> xyz_coord)   
    #     peak_idx = peak_local_max(imgOut[:,:, 0],
    #                               min_distance=int(np.round(d_min)),
    #                               threshold_abs=I_threshold,
    #                                # exclude_border=int(np.round(1*scale)) # ???
    #                               )
    # else:
    #     #Compute: Local Maxima at  the i-th scale (peak_idx -> xyz_coord)   
    #     peak_idx = peak_local_max(imgOut,
    #                               min_distance=int(np.round(d_min)),
    #                               threshold_abs=I_threshold,
    #                                # exclude_border=int(np.round(1*scale)) # ???
    #                               )
        
 
    # #Op1: Get Intesity Values
    # #Note: this have a mistmatch with the coordinates 
    # peak_mask = np.zeros_like(imgOut, dtype=bool)
    # peak_mask[tuple(peak_idx.T)] = True
    
    # I_Raw = imgIn_Raw[peak_mask]
    # I_Pro = imgIn_Pro[peak_mask]
    # I_DoG  = imgOut[peak_mask]
    
    #Op2: Get Intesity Values
    idx = np.ndarray.tolist(peak_idx.T)  
    
    I_Raw = imgIn_Raw[tuple(idx)]
    I_Pro = imgIn_Pro[tuple(idx)]
    I_DoG  = imgOut[tuple(idx)] 
    

    # if imgOut.shape[2]==1:
    #     colZeros = np.zeros((peak_idx.shape[0], 1))
    #     peak_idx = np.hstack((peak_idx, colZeros))
    
    # print()
    # print('scale', scale)
    # print('imgIn.shape', imgIn_Raw.shape)
    # print('imgIn (Min, Max)', imgIn_Raw.min(), imgIn_Raw.max())
    # print('imgOut.shape', imgOut.shape)
    # print('imgOut (Min, Max)', imgOut.min(), imgOut.max())
    
    # print()
    # print('Hola')
    # print('peak_idx \n', peak_idx) 
    # sys.exit()
    
    # xyz_coord = peak_local_max(imgOut, min_distance=int(d_min), indices=True, threshold_abs=threshold)
    # maxMatrix = peak_local_max(imgOut, min_distance=int(d_min), indices=False, threshold_abs=threshold) 

    return peak_idx, I_Raw, I_Pro, I_DoG

    

#1.3) Compute the Maximum along Spatial Scales
#Note: The term "Point" refers to each Detected Local Maxima at each Scale (i.e. a putative detected cell)
def compute_ScaleMaximumMS(df, scales, t0=0):
    start = time.time() - t0
    
    #The algorithm starts the analysis from the bigger Scale   
    scales = np.flip(scales)
    
    #Routine to get the Maxima of Local Maxima along Scales
    for i in range(0, scales.shape[0]): 
        #1) Extract all Detected Point from the i-th scale 
        currentScale = scales[i]
        dfS = df.loc[(df['S'] == currentScale)] 
        # print('') 
        # print('-----------------')
        # print('Current Scale=', currentScale)  
#        if currentScale == scales[-1]:
#            print dfS
#            n_cells = dfS.shape[0]
#            for c in range(0, n_cells):
#                dfS.iloc[c]
#            jaja
            
        
        for j in dfS.index: 
            #2) Extract the j-th Detected Point at the i-th Scale               
            p = dfS.loc[j]
            x, y, z = p['X'], p['Y'], p['Z'] 
#            print('')        
#            print('Point Index=', j) 
                  
                  
            
            # ??? 3) ??? Find Detected Points Along Lower Scales to check if...
            # kr = 0.10
            # kr = 0.25
            # kr = 0.5
            kr = 1.0
            # kr = 1.5
            # kr = 2.0
            r = kr*currentScale + 1 
            I = p['I_DoG']                       
            currentLowerScales = scales[i+1:]
            # print('') 
            # print('Current Point \n', p) 
            # print (currentScale)
            # print(currentLowerScales)
            # print (p) 
            for k in range(0, currentLowerScales.shape[0]):
                #3.1) Extract all Detected Point from the k-th lower scale (where k-th<i-th) 
                currentLowerScale = currentLowerScales[k]
                dfS_low = df.loc[(df['S'] == currentLowerScale)]  
                
                #3.2) Compute the Euclidian Distance between the following points:
                #       a) The j-th Detected Point at the i-th Scale and...
                #       b) All Detected Points from the k-th lower scale (where k-th<i-th)
                dr = np.sqrt((dfS_low['X'] - x)**2 + (dfS_low['Y'] - y)**2 + (dfS_low['Z'] - z)**2)
                
                
                # r = kr*currentLowerScale + 1
                # print()
                # print('r_min')
                # print(r)
                
                #3.3) Extract the Detected Points of the k-th lower scale whose...
                #     Euclidean distance from the Detected Point of the i-th upper scale 
                #     are equal or lower than the radius of an Spherical object at the i-th upper scale                            
                maskBool = (dr<=r)
                ix = maskBool.index
                ix = ix[maskBool]   
                df_pts = df.loc[ix] 
                

                
                #3.4) Check if the Local Maxima of the i-th current scale is greater than...
                #     all the Local Maxima of the k-th lower scale that...
                #     are inside the volume of a Spherical object at the i-th upper scale  
                myBool = I>df_pts['I_DoG'].values
                 
                # print()
                # print('I', I)
                # print('I_pts', df_pts['I_DoG'].values)
                # print(myBool)

                
                #A) The local maxima of the current state is the maximum across scale
                if myBool.sum()>0:
                    #Remove from the DataFrame all Local Maxima of the k-th lower scale 
                    df = df.drop(ix)
                
                #B) The local maxima of the current state is not the maximum across scales
                else:  
                    #Remove from the DataFrame the Local Maxima of the i-th upper scale 
                    df = df.drop(j)
                    break
                
#                print('')        
#                print('Current Lower Scale=', currentLowerScale)         

    stop = time.time() - t0
    return df, start, stop


#1.4) Compute the Number of times that the same Cell is detected at several Spatial Scales
def compute_ScaleCount(df_All, df_Cells, t0=0):
    start = time.time() - t0
    
    nCells = df_Cells.shape[0]
    N = np.zeros(nCells)
    for i in range(0,nCells):
        pt = df_Cells.iloc[i]
        x0, y0, z0, R = pt['X'], pt['Y'], pt['Z'], pt['S'] 
        dr = np.sqrt((df_All['X'] - x0)**2 + (df_All['Y'] - y0)**2 + (df_All['Z'] - z0)**2)
        
        #Number of pts Inside the Sphere of the Object
        myBool = dr<1.0*R
        N[i] = myBool.sum()

        
    df_Cells['N'] = N
    stop = time.time() - t0
    return df_Cells, start, stop


#==============================================================================
#   Remove False Positives 
#==============================================================================
#Based on the Number of Detections Along Spatial Scales  
def remove_FalseCells_with_LowScaleCount(df_Cells, scaleCountThreshold, t0=0):
    start = time.time() - t0
    
    maskBool = df_Cells['N']<scaleCountThreshold
    ix = maskBool.index
    ix = ix[maskBool] 
    df_Cells = df_Cells.drop(ix)
    
    stop = time.time() - t0
    return df_Cells, start, stop

#Based on the Cell Intensity
def remove_FalseCells_with_LowIntensity(df_Cells, I_threshold, mode='none', t0=0):    
    start = time.time() - t0
    if mode=='absolute':
        maskBool1 = df_Cells['I_Raw']<I_threshold
#        maskBool = df_Cells['I']<I_threshold
        
        # maskBool2 = df_Cells['dI']<=0.20
#        maskBool2 = (df_Cells['I']/df_Cells['I0'])<=0.3
        
        # maskBool = maskBool1|maskBool2
        maskBool = maskBool1
#        print(maskBool1)
#        print(maskBool2)
#        print(maskBool)
        
    elif mode=='relative':
        if (I_threshold>0)&(I_threshold<=1):
            maskBool = df_Cells['I']<I_threshold*(df_Cells['I'].max())
        else:
            print('')
            print('remove_LowIntensityDetections')
            print('The "I_threhold" argument must be within (0...1]')
    
    elif mode=='none':
            print('')
            print('remove_LowIntensityDetections')
            print('The "mode" argument is "None"')
    else:
            print('')
            print('remove_LowIntensityDetections')
            print('The "mode" argument must be "absolute" or "relative"')
    
    ix = maskBool.index
    ix = ix[maskBool] 
    df_Cells = df_Cells.drop(ix)
    
    stop = time.time() - t0
    return df_Cells, start, stop




#==============================================================================
# Other function
#==============================================================================

# def get_spatialScaleRangeInPixels(r_min_um, r_max_um, resolution):
#     res_min = np.min(resolution)
#     r_min_px = int(np.floor(r_min_um/res_min))
#     r_max_px = int(np.ceil(r_max_um/res_min))      
#     scales = np.arange(r_min_px, r_max_px + 1 , dtype=np.float)
#     return scales

def get_spatialScaleRangeInPixels(r_min_um, r_max_um, voxelSize_um, n_scales=None):
    smallest_voxelSize_um = np.min(voxelSize_um)
    r_min_px = int(np.floor(r_min_um/smallest_voxelSize_um))
    r_max_px = int(np.ceil(r_max_um/smallest_voxelSize_um))      
    if not n_scales:
        n_scales = int((r_max_px + 1) - r_min_px)
    # scales = np.linspace(r_min_px, r_max_px, n_scales) # ????
    scales = np.arange(r_min_px, r_max_px + 1)         # ????
    # scales = scales[scales>2]
    return scales
#==============================================================================
#     
#==============================================================================
    
if __name__== '__main__':

    df = pd.DataFrame()

    v = [[1, 2, 3, 4, 5]]
    df = df.append(v)

    v = [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]
    df = df.append(v)
    
    print (df)
#==============================================================================
#     
#==============================================================================
#    columns = ['S','X', 'Y', 'Z', 'I']
#    df = pd.DataFrame(columns=columns)
#    
#    v = [[1, 2, 3, 4, 5]]
#    v = [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]
#    sf = pd.Series(v)
#    df = df.append(sf, ignore_index=True)
#    print df
#    data = np.column_stack(v)
    
#    np.vstack([x, y, z, I])
    
    data = np.column_stack((v))
    print (data)





#==============================================================================
# Draft
#==============================================================================

#def remove_LowIntensityDetections(df_Cells, I_threshold, mode='relative'):
#    
#    if mode=='absolute':
#        maskBool = df_Cells['I']<I_threshold
#        
#    elif mode=='relative':
#        if (I_threshold>0)&(I_threshold<=1):
#            maskBool = df_Cells['I']<I_threshold*(df_Cells['I'].max())
#        else:
#            print('')
#            print('remove_LowIntensityDetections')
#            print('The "I_threhold" argument must be within (0...1]')
#    else:
#            print('')
#            print('remove_LowIntensityDetections')
#            print('The "mode" argument must be "absolute" or "relative"')
#    
#    ix = maskBool.index
#    ix = ix[maskBool] 
#    df_Cells = df_Cells.drop(ix)
#    return df_Cells

#==============================================================================
# 
#==============================================================================
##Based on the Cell Intensity
#def remove_FalseCells_with_LowIntensity(df_Cells, intensityRatio):    
#
#    if (intensityRatio>=0)&(intensityRatio<=1):
#        maskBool = df_Cells['I']<intensityRatio*(df_Cells['I'].max())
##        maskBool = df_Cells['I0']<intensityRatio*(df_Cells['I0'].max())
#    else:
#        print('')
#        print('remove_LowIntensityDetections')
#        print('The "I_threhold" argument must be within (0...1]')
#
#    
#    ix = maskBool.index
#    ix = ix[maskBool] 
#    df_Cells = df_Cells.drop(ix)
#    return df_Cells
#==============================================================================
# 
#==============================================================================

#        #Op3:
#        s = [scale]
#        Fx = get_DoG(s, rS)
#        
#        imgX = cv2.filter2D(imgIn, -1, Fx, borderType=0) 
#    
#        imgY = cv2.filter2D(np.transpose(imgIn, axes=(1,0,2)), -1, Fx, borderType=0)
#        imgY = np.transpose(imgY, axes=(1,0,2))
#        
#        imgZ = cv2.filter2D(np.transpose(imgIn, axes=(0,2,1)), -1, Fx, borderType=0)
#        imgZ = np.transpose(imgZ, axes=(1,0,2))
#        imgOut = np.sqrt(imgX**2+imgY**2+imgZ**2)

# =============================================================================
# fft filtering computation
# =============================================================================

        #Specific Initialization due to the Convolution in the Frequency Domain
        # imgIn_fft = np.fft.fftn(imgIn)
    
        #Compute: Convolve an 3DImage with a 3D Filter (Computing Mode: Frequency Domain)        
        #Op1        
        # Fxyz_fft =  np.fft.fftn(Fxyz, (imgIn_fft.shape))
        # imgOut_fft = imgIn_fft*np.abs(Fxyz_fft)
        # imgOut =     np.fft.ifftn(imgOut_fft)
        # imgOut = imgOut.real



