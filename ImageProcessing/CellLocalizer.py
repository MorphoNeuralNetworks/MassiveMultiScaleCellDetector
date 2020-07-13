# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:30:28 2020

@author: pc
"""
import numpy as np
import pandas as pd
import time


from ImageFilters import get_DoG
from skimage.feature import peak_local_max

from scipy import signal
#import cv2


#==============================================================================
#Note: Maximum (singular); Maxima (plural)
#Note: Scale is given as the Radius of the Sphere that encolses the 3D-Object
#==============================================================================

#==============================================================================
# Main Routine
#==============================================================================

def run_CellLocalizer(imgIn, scales, rS, rV):
    
    #1.1) Compute the Second Derivative of the Input Image at several Spatial Scales
    #Note: The second derivative is computed through a DoG Filter  
    imgDoGMS, dt1 = compute_SecondDerivativeMS(imgIn, scales, rS, rV)
   
    #1.2) Compute the Local Maxima at each Spatial Scale 
    df_All, dt2 = compute_SpatialMaximaMS(imgIn, imgDoGMS, scales)                                                          
    
    #1.3) Compute the Maximum along Spatial Scales
    df_Cells, dt3 = compute_ScaleMaximumMS(df_All, scales)
    
    #1.4) Compute the Number of times that the same Cell is detected at several Spatial Scales
    df_Cells, dt4 = compute_ScaleCount(df_All, df_Cells)
    
    #Computing Times
    dt = np.asarray([dt1, dt2, dt3, dt4])
    
    return imgDoGMS, df_All, df_Cells, dt


#==============================================================================
#   Subrutines
#==============================================================================

#1.1) Compute the Second Derivative of the Input Image at several Spatial Scales
def compute_SecondDerivativeMS(imgIn, scales, rS, rV):
    
    #General Initialization
    t0 = time.time() 
    ns = scales.shape[0]
    imgDoGMS = np.empty(ns, dtype=object)

    
    #Specific Initialization due to the Convolution in the Frequency Domain
#    imgIn_fft = np.fft.fftn(imgIn)

    for i in range(0,ns): 
        #Get the i-th scale 
        scale = scales[i]    
        
        #Get the 3D-DoG Filter at the i-th scale (Isotropic Filter)
        s = [scale, scale, scale]
        Fxyz = get_DoG(s, rS, rV)
        
        #Compute: Convolve an 3DImage with a 3D Filter (Computing Mode: Frequency Domain)        
        #Op1        
#        Fxyz_fft =  np.fft.fftn(Fxyz, (imgIn_fft.shape))
#        imgOut_fft = imgIn_fft*np.abs(Fxyz_fft)
#        imgOut =     np.fft.ifftn(imgOut_fft)
#        imgOut = imgOut.real
        
        #Op2
        imgOut = signal.convolve(imgIn, Fxyz, "same")
        
    
#        #Visualize
#        from matplotlib import pyplot as plt
#        import matplotlib.cm as cm
#        plt.imshow(imgOut[:, :,imgIn.shape[2]//2], cm.Greys_r, interpolation='nearest')
#        plt.show()

        #Store the Output at the i-th scale        
        imgDoGMS[i] = imgOut         
        

    dt = time.time()  - t0
    return imgDoGMS, dt

#1.2) Compute the Local Maxima at all Spatial Scales     
def compute_SpatialMaximaMS(imgIn, imgDoGMS, scales, I_threshold=0.0):
    #General Initialization
    t0 = time.time() 
    ns = scales.shape[0]
    table = [] 
    
    for i in range(0,ns):
        #Get the Second Derivative Scale
        imgOut = imgDoGMS[i]
        scale = scales[i]
        
        #Compute: Local Maxima of hte 
        xyz_coord, I0, I = compute_SpatialMaxima(imgIn, imgOut, scale)
        
        xyz_coord[:,[0, 1]] = xyz_coord[:,[1, 0]]        
        
        #Store the                  
        S = scale*np.ones(I.shape[0])
        #data = np.column_stack((S, xyz_coord, I))
        data = np.column_stack((xyz_coord, S, I0, I))
        if not len(table):
            table = data
        else:       
            table = np.vstack((table, data))
    
    columns = ['X', 'Y', 'Z', 'S', 'I0', 'I']
    
    df = pd.DataFrame(table, columns=columns)
    dt = time.time()  - t0
    return df, dt
    
    
#1.2.1) Compute the Local Maxima at a single Spatial Scale   
def compute_SpatialMaxima(imgIn, img3D, scale, threshold=0.0, k=1.0):
    #Get the radius of the Spheric Volume over which compute the Maximum 
    d_min = k*scale
    
    #Compute: Local Maxima at  the i-th scale    
    xyz_coord = peak_local_max(img3D, min_distance=int(d_min), indices=True, threshold_abs=threshold)
    
    #Get: Intesity Values
    maxMatrix = peak_local_max(img3D, min_distance=int(d_min), indices=False, threshold_abs=threshold) 
    I = img3D[maxMatrix]
    I0 = imgIn[maxMatrix]
    
    return xyz_coord, I0, I

    

#1.3) Compute the Maximum along Spatial Scales
#Note: The term "Point" refers to each Detected Local Maxima at each Scale (i.e. a putative detected cell)
def compute_ScaleMaximumMS(df, scales):
    t0 = time.time()
    
    #The algorithm starts the analysis from the bigger Scale   
    scales = np.flip(scales)
    
    #Routine to get the Maxima of Local Maxima along Scales
    for i in range(0, scales.shape[0]): 
        #1) Extract all Detected Point from the i-th scale 
        currentScale = scales[i]
        dfS = df.loc[(df['S'] == currentScale)] 
#        print('') 
#        print('-----------------')
#        print('Current Scale=', currentScale)         
        
        for j in dfS.index: 
            #2) Extract the j-th Detected Point at the i-th Scale               
            p = dfS.loc[j]
            x, y, z = p['X'], p['Y'], p['Z'] 
#            print('')        
#            print('Point Index=', j) 
                        
            
            #3) Find Detected Points Along Lower Scales to check if...
            r = currentScale + 1 
#            r = 1.5*currentScale + 1 
            I = p['I']                       
            currentLowerScales = scales[i+1:]
            for k in range(0, currentLowerScales.shape[0]):
                #3.1) Extract all Detected Point from the k-th lower scale (where k-th<i-th) 
                currentLowerScale = currentLowerScales[k]
                dfS_low = df.loc[(df['S'] == currentLowerScale)]  
                
                #3.2) Compute the Euclidian Distance between the following points:
                #       a) The j-th Detected Point at the i-th Scale and...
                #       b) All Detected Points from the k-th lower scale (where k-th<i-th)
                dr = np.sqrt((dfS_low['X'] - x)**2 + (dfS_low['Y'] - y)**2 + (dfS_low['Z'] - z)**2)
                
                
                #3.3) Extract the Detected Points of the k-th lower scale whose...
                #     Euclidean distance from the Detected Point of the i-th upper scale 
                #     are equal or lower than the radius of an Spherical object at the i-th upper scale                            
                maskBool = (dr<=r)
                ix = maskBool.index
                ix = ix[maskBool]   
                df_pts = df.loc[ix] 
                
                #3.4) Chek if the Local Maxima of the i-th current scale is greater than...
                #     all the Local Maxima of the k-th lower scale that...
                #     are inside the volume of a Spherical object at the i-th upper scale  
                myBool = I>df_pts['I'].values
                                
            
                #A) The local maxima of the current state is the maximum across scale
                if myBool.sum()>0:
                    #Remove from the DataFrame all Local Maxima of the k-th lower scale 
                    df = df.drop(ix)
                
                #B) The local maxima of the current state is not the maximum across scales
                else:  
                    #Remove from the DataFrame the Local Maxima of the i-th current scale 
                    df = df.drop(j)
                    break
                
#                print('')        
#                print('Current Lower Scale=', currentLowerScale)         

    dt = time.time()  - t0
    return df, dt


#1.4) Compute the Number of times that the same Cell is detected at several Spatial Scales
def compute_ScaleCount(df_All, df_Cells):
    t0 = time.time()
    
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
    dt = time.time()  - t0
    return df_Cells, dt


#==============================================================================
#   Remove False Positives 
#==============================================================================
#Based on the Number of Detections Along Spatial Scales  
def remove_FalseCells_with_LowScaleCount(df_Cells, scaleCountThreshold):
  
    maskBool = df_Cells['N']<scaleCountThreshold
    ix = maskBool.index
    ix = ix[maskBool] 
    df_Cells = df_Cells.drop(ix)
    
    return df_Cells

#Based on the Cell Intensity
def remove_FalseCells_with_LowIntensity(df_Cells, I_threshold, mode='absolute'):    

    if mode=='absolute':
        maskBool = df_Cells['I']<I_threshold
        
    elif mode=='relative':
        if (I_threshold>0)&(I_threshold<=1):
            maskBool = df_Cells['I']<I_threshold*(df_Cells['I'].max())
        else:
            print('')
            print('remove_LowIntensityDetections')
            print('The "I_threhold" argument must be within (0...1]')
    else:
            print('')
            print('remove_LowIntensityDetections')
            print('The "mode" argument must be "absolute" or "relative"')
    
    ix = maskBool.index
    ix = ix[maskBool] 
    df_Cells = df_Cells.drop(ix)
    return df_Cells

#==============================================================================
# Other function
#==============================================================================

def get_spatialScaleRangeInPixels(r_min_um, r_max_um, resolution):
    res_min = np.min(resolution)
    r_min_px = int(np.floor(r_min_um/res_min))
    r_max_px = int(np.ceil(r_max_um/res_min))      
    scales = np.arange(r_min_px, r_max_px + 1 , dtype=np.float)
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
    
    print df
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
    print data





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







