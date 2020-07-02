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
from scipy.signal import fftconvolve
#    if (isIsotropic==True) & (mode=='fft'):
#        run_Isotropic_fft(imgIn3D, scales, rS, rV, Imin)
#def run_Isotropic_fft (imgIn3D, scales, rS, rV, Imin):  



#==============================================================================
# Compute the Local Maxima along Scales 
#
#Note: Maximum (singular); Maxima (plural)
#Note: Scale is given as the Radius of the Sphere that encolses the 3D-Object
#==============================================================================
#def compute_LocalMaximaAlongAllScales(imgIn3D, scales, rS, rV, Imin, isIsotropic=True, mode='fft')
#    if (isIsotropic==True) & (mode=='fft'):
#        print('Isotropic and fft Convolution')
def compute_LocalMaximaAlongAllScales(imgIn2D, scales, rS, rV=1.0, I_threshold=0.0):
    t0 = time.time()    
    
    #General Initialization
    ns = scales.shape[0]
    imgOutMS = np.empty(ns, dtype=object)
    Fxy = np.empty(ns, dtype=object)
    table = [] 
    
    #Specific Initialization due to the Convolution in the Frequency Domain
    imgIn_fft = np.fft.fftn(imgIn2D)

    for i in range(0,ns): 
        #Get the i-th scale 
        scale = scales[i]    
        
        #Get the 3D-DoG Filter at the i-th scale (Isotropic Filter)
        s = [scale, scale]
        Fxy[i] = get_DoG(s, rS, rV, a=1.0)
        
        #Compute: Convolve an 3DImage with a 3D Filter (Computing Mode: Frequency Domain)
        Fxy_fft =  np.fft.fftn(Fxy[i], (imgIn_fft.shape))
        imgOut_fft = imgIn_fft*np.abs(Fxy_fft)
        imgOut =     np.fft.ifftn(imgOut_fft)
        imgOut = imgOut.real
        
#        imgOut = fftconvolve(imgIn2D, Fxy, mode='same')

#        print('')
#        print(scale)
#        print('Gmax')
#        Fxy_fft = np.fft.fftshift(Fxy_fft)
#        Fxy_fft = np.abs(Fxy_fft) 
#        print(Fxy_fft.max())
        #Store the Output at the i-th scale        
        imgOutMS[i] = imgOut         
        
        #Compute: Local Maxima of hte 
        xy_coord, I = compute_LocalMaxima(imgOut, scale, threshold=I_threshold, k=1.0)

        #Store the                  
        S = scale*np.ones(I.shape[0])
        data = np.column_stack((xy_coord, S, I))
        if not len(table):
            table = data
        else:       
            table = np.vstack((table, data))
    
    columns = ['X', 'Y',  'S', 'I']
    
    df = pd.DataFrame(table, columns=columns)
    dt = time.time()  - t0
    
    return df, imgOutMS, Fxy, dt
  
  
  
#Peaks are the local maxima in a region of
#  2 * min_distance + 1 (i.e. peaks are separated by at least min_distance).

#If there are multiple local maxima with identical pixel intensities
#inside the region defined with min_distance,
#the coordinates of all such pixels are returned.

def compute_LocalMaxima(img2D, scale, threshold=0.0, k=1.0):
    #Get the radius of the Spheric Volume over which compute the Maximum 
    d_min = np.ceil(0.5*k*scale)
    d_min = scale - 1
    
    #Compute: Local Maxima at  the i-th scale    
    xy_coord = peak_local_max(img2D, min_distance=int(d_min), indices=True, threshold_abs=threshold)
    
    #Get: Intesity Values
    maxMatrix = peak_local_max(img2D, min_distance=int(d_min), indices=False, threshold_abs=threshold) 
    I = img2D[maxMatrix]
    
    return xy_coord, I

    
#==============================================================================
#   Rotine to compute the Maxima of Local Maxima along all Spatial Scales
#==============================================================================
#Note: The term "Point" refers to each Detected Local Maxima at each Scale (i.e. a putative detected cell)
def compute_MaximaOfLocalMaximaAlongAllScales(df, scales):
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
            x, y= p['X'], p['Y']
#            print('')        
#            print('Point Index=', j) 
                        
            
            #3) Find Detected Points Along Lower Scales to check if 
            r = currentScale + 1 
            r = 1.5*currentScale + 1 
            I = p['I']                       
            currentLowerScales = scales[i+1:]
            for k in range(0, currentLowerScales.shape[0]):
                #3.1) Extract all Detected Point from the k-th lower scale (where k-th<i-th) 
                currentLowerScale = currentLowerScales[k]
                dfS_low = df.loc[(df['S'] == currentLowerScale)]  
                
                #3.2) Compute the Euclidian Distance between the following points:
                #       a) The j-th Detected Point at the i-th Scale and...
                #       b) All Detected Points from the k-th lower scale (where k-th<i-th)
                dr = np.sqrt((dfS_low['X'] - x)**2 + (dfS_low['Y'] - y)**2 )
                
                
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




















