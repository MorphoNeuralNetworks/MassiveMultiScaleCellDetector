# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:01:14 2020

@author: pc
"""

import numpy as np
from scipy import signal
import time


from ImageFilters import get_DxGaussian, get_Gaussian


#==============================================================================
# Main Routine
#==============================================================================

def run_Tensor(imgDoGMS, df_Cells, scales):
    
    #Compute the Tensor only for those Spatial Scales where cells were detected    
    ss = np.unique(df_Cells['S'].values)
    n_scales = ss.shape[0]
    imgMS = np.empty(n_scales, dtype=object)
    for i in range(0, n_scales):
        ix = np.where(scales==ss[i])[0][0]
        imgMS[i] = imgDoGMS[ix]
    
    TensorMS, dt1 = compute_TensorMS(imgMS, ss)
    df_Cells, dt2 = compute_PtsTensorMetrics(TensorMS, df_Cells.copy(), ss)

    dt = np.asarray([dt1, dt2])
    return df_Cells, dt
#==============================================================================
# 
#==============================================================================
def compute_TensorMS(imgDoGMS, scales):
    
    t0 = time.time()
    ns = scales.shape[0]
    TensorMS = np.empty(ns, dtype=object)
    for i in range(0, ns):
        s = scales[i]
        img = imgDoGMS[i]    
        TensorMS[i] = compute_Tensor(img, s)

    dt = time.time()  - t0
    return TensorMS, dt
    
def compute_Tensor(img, s):
    #Create
    s = [s, s, s]
    
    #Scale Control
#    k = 0.05
#    k = 0.15
    k = 0.25
#    k = 0.50
#    k = 1.00
#    k = 2.00
    
    #First Derivative of a Gaussian 
    Fx = get_DxGaussian(s, a=k)    
    Fy = np.transpose(Fx, (1, 0, 2))
    Fz = np.transpose(Fx, (0, 2, 1))

    
    #Gaussian 
    G = get_Gaussian(s, a=1.0*k)
    
    #Partial Derivatives
    Dx = signal.convolve(img, Fx, "same")
    Dy = signal.convolve(img, Fy, "same")
    Dz = signal.convolve(img, Fz, "same")
    
    #Tensor Components
    Dxx = Dx**2
    Dyy = Dy**2
    Dzz = Dz**2
    Dxy = Dx*Dy
    Dxz = Dx*Dz
    Dyz = Dy*Dz
    
    #Smooth the Partial Derivatives
    Dxx = signal.convolve(Dxx, G, "same") 
    Dyy = signal.convolve(Dyy, G, "same") 
    Dzz = signal.convolve(Dzz, G, "same") 
    Dxy = signal.convolve(Dxy, G, "same") 
    Dxz = signal.convolve(Dxz, G, "same") 
    Dyz = signal.convolve(Dyz, G, "same")
  

    return [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
 
 
#==============================================================================
# 
#==============================================================================
def compute_PtsTensorMetrics(TensorMS, dfMS, scales):
    t0 = time.time()
 
    n = dfMS.shape[0]
    v_orientation = np.zeros((n,3))
    v_anisotropy = np.zeros(n)
    v_tubularity = np.zeros(n)
    v_disk = np.zeros(n)
    
    #Select Scales
    ss = np.unique(dfMS['S'].values)
    n_scales = ss.shape[0]
    k = 0
    for i in range(0, n_scales):
        s = ss[i]
        ix = np.where(scales==s)[0][0]
        [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = TensorMS[ix]
        
        #Select Points in the Scale       
        dfS = dfMS.loc[(dfMS['S'] == s)]  
        n_pts = dfS.shape[0]
        dfS = dfS.astype(int)
        x, y, z = dfS['Y'].values, dfS['X'].values, dfS['Z'].values
        for j in range(0, n_pts):
            x0, y0, z0 = x[j], y[j], z[j]
            T = np.asarray([[Dxx[x0,y0,z0], Dxy[x0,y0,z0], Dxz[x0,y0,z0]],
                                [Dxy[x0,y0,z0], Dyy[x0,y0,z0], Dyz[x0,y0,z0]],
                                [Dxz[x0,y0,z0], Dyz[x0,y0,z0], Dzz[x0,y0,z0]]])
    
            v_orientation[k,:], v_anisotropy[k], v_tubularity[k], v_disk[k] = compute_TensorMetrics(T)
            k = k + 1            
            
    dfMS['Vx'] = v_orientation[:,0]
    dfMS['Vy'] = v_orientation[:,1]
    dfMS['Vz'] = v_orientation[:,2]
    
    dfMS['Ani'] = v_anisotropy
    dfMS['Tub'] = v_tubularity
    dfMS['Disk'] = v_disk
    
    dt = time.time()  - t0
    return dfMS, dt


def compute_TensorMetrics(T):  
    
    
    #Eigenvalues
    eigVal, eigVec = np.linalg.eig(T) 

    #Normalization    
    k_norm = 1.0/np.sqrt(eigVal[0]**2+eigVal[1]**2+eigVal[2]**2)
    eigVal = k_norm*eigVal

    #Anisotropy
    eigValPairs = np.asarray([((eigVal[0]-eigVal[1])/(eigVal[0]+eigVal[1]))**2,
                              ((eigVal[0]-eigVal[2])/(eigVal[0]+eigVal[2]))**2,
                              ((eigVal[1]-eigVal[2])/(eigVal[1]+eigVal[2]))**2])         
    anisotropy = np.sqrt(np.sum(eigValPairs))
    
    #Tubularity
    p1, p2, p3 = np.sort(eigVal)[::-1]
    tubularity =  (np.sqrt(p1**2 + p2**2))/p3 - np.sqrt(2)
    
    #Disc
    disk = p1/(np.sqrt(p2**2 + p3**2)) - 1.0/np.sqrt(2)

    
    
    #Orientation
    ix = np.argmin(eigVal)
    v = eigVec[:,ix]
    orientation = v/(np.sqrt(v[0]**2 + v[1]**2 + v[2]**2))
    
      
    return orientation, anisotropy, tubularity, disk


 
if __name__== '__main__':
    pass



#==============================================================================
# 
#==============================================================================
#def compute_PtsTensorMetrics(TensorMS, dfMS, scales):
#     
#    n = dfMS.shape[0]
#    v_orientation = np.zeros((n,3))
#    v_anisotropy = np.zeros(n)
#    v_tubularity = np.zeros(n)
#    
#    #Select Scales
#    ss = np.unique(dfMS['S'].values)
#    n_scales = ss.shape[0]
#    k = 0
#    for i in range(0, n_scales):
#        s = ss[i]
#        ix = np.where(scales==s)[0][0]
#        [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = TensorMS[ix]
#        
#        #Select Points in the Scale
#        mask = ss==s
#        dfS = dfMS[mask]
#        
#        
#        n_pts = dfS.shape[0]
#        dfS = dfS.astype(int)
#        x, y, z = dfS['Y'].values, dfS['X'].values, dfS['Z'].values
#        for j in range(0, n_pts):
#            x0, y0, z0 = x[i], y[i], z[i]
#            T = np.asarray([[Dxx[x0,y0,z0], Dxy[x0,y0,z0], Dxz[x0,y0,z0]],
#                                [Dxy[x0,y0,z0], Dyy[x0,y0,z0], Dyz[x0,y0,z0]],
#                                [Dxz[x0,y0,z0], Dyz[x0,y0,z0], Dzz[x0,y0,z0]]])
#    
#            v_orientation[k,:], v_anisotropy[k], v_tubularity[k] = compute_TensorMetrics(T)
#            k = k + 1            
#            
#    dfMS['Vx'] = v_orientation[:,0]
#    dfMS['Vy'] = v_orientation[:,1]
#    dfMS['Vz'] = v_orientation[:,2]
#    
#    dfMS['Anisotropy'] = v_anisotropy
#    dfMS['Tubularity'] = v_tubularity
#    
#    return dfMS


#==============================================================================
# 
#==============================================================================
#def compute_PointTensorMetrics(TensorMS, scales, ptsMS, s):
#     
#    #Select Scales
#    ss = np.unique(ptsMS[3])
#    n_scales = ss.shape[0]
#    for i in range(0, n_scales):
#        s = ss[i]
#        ix = np.where(scales==s)[0][0]
#        [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = TensorMS[ix]
#        
#        #Select Points in the Scale
#        mask = ss==s
#        pts = ptsMS[mask]
#        n_pts = pts.shape[0]
#        for j in range(0, n_pts):
#            x0, y0, z0 = pts[i] 
#            T = np.asarray([[Dxx[x0,y0,z0], Dxy[x0,y0,z0], Dxz[x0,y0,z0]],
#                                [Dxy[x0,y0,z0], Dyy[x0,y0,z0], Dyz[x0,y0,z0]],
#                                [Dxz[x0,y0,z0], Dyz[x0,y0,z0], Dzz[x0,y0,z0]]])
#    
#    
#        
#    n_pts = pts.shape[0]
#    
#
#    
#    #Compute Metrics
#    eigVal, eigVec = np.linalg.eig(T)

#==============================================================================
#  
#==============================================================================
