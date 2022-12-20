# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:01:14 2020

@author: pc
"""

import numpy as np
from scipy import signal
import time


#from ImageFilters import get_DxGaussian, get_Gaussian
from ImageProcessing.ImageFilters import get_DxGaussian, get_Gaussian
        
import os
import psutil
import itertools

def sphere(n):
    n = (np.ceil(n)).astype(int)
    
    struct = np.zeros((2 * n + 1, 2 * n + 1, 2 * n + 1))
    x, y, z = np.indices((2 * n + 1, 2 * n + 1, 2 * n + 1))
    mask = (x - n)**2 + (y - n)**2 + + (z - n)**2 <= n**2
    struct[mask] = 1
    return struct


#==============================================================================
# Main Routine
#==============================================================================

def run_Tensor(imgIn, imgDoGMS, df_Cells, scales, t0=0):
    
    #Compute the Tensor only for those Spatial Scales where cells were detected  
    ss = np.unique(df_Cells['S'].values)
#    ss = 1.0*scales
    n_scales = ss.shape[0]
    imgMS = np.empty(n_scales, dtype=object)
    for i in range(0, n_scales):
        ix = np.where(scales==ss[i])[0][0]
        imgMS[i] = imgDoGMS[ix]
        # ????? Bypassing
        # imgMS[i] =  imgIn 
    
    # imgMS = imgDoGMS
    #Compute Tensor
    TensorMS, start, stop = compute_TensorMS(imgMS, ss, t0=t0)
    op1 = [start, stop]
    

#    p = psutil.Process(os.getpid())
#    M = p.memory_info()[0] # in bytes 
#    print('')
#    print('Memory Usage: run_Tensor')
#    print (1.0*M/10**9)  
#    print(p.memory_info())
    
    #Compute Tensor Metrics
    df_Cells, start, stop = compute_PtsTensorMetrics(TensorMS, df_Cells.copy(), ss, t0=t0)
    op2 = [start, stop]
    
    op = [op1, op2]
    return df_Cells, op
#==============================================================================
# 
#==============================================================================
def compute_TensorMS(imgDoGMS, scales, t0=0):
    
    start = time.time() - t0
    ns = scales.shape[0]
    TensorMS = np.empty(ns, dtype=object)
    for i in range(0, ns):
        s = scales[i]
        img = imgDoGMS[i]  
        
        # # # ???? Opening
        # from skimage.morphology import opening
        # img = opening(img, sphere(n=s/2.0))
        
        # # ??? Smoothing
        # s_Gauss = np.array([s,s,s])
        # s_Gauss = s_Gauss/1
        # Fxyz = get_Gaussian(s=s_Gauss)
        # img = signal.convolve(img, Fxyz, "same")
        
        
        TensorMS[i] = compute_Tensor(img, s)

    stop = time.time()- t0
    return TensorMS, start, stop
    
def compute_Tensor(img, s, k=1.0):
    #Create
    s = np.array([s, s, s])
    
    #Scale Control
#    k = 0.05
#    k = 0.15
    # k = 0.15    #more acurate when there is no noise
    # k = 0.50
    # k = 1.00
    # k = 2.00
    # k = 2.00
    # s_gauss = [s_gauss, s_gauss, s_gauss]
    # s_Dx = [s_Dx, s_Dx, s_Dx]
    
    #Filter: High Pass Filter (First Order Derivative of a Gaussian)
    s_Dx = s
    Fx = get_DxGaussian(s_Dx, a=k)    
    Fy = np.transpose(Fx, (1, 0, 2))
    Fz = np.transpose(Fx, (0, 2, 1))

    #Image : First Order Partial Derivatives 
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
    
    #Filter: Low Pass Filter (Gaussian) 
    # s_gauss = 2*s
    # s_gauss = s
    s_gauss = s/2.0
    # s_gauss = 0.5*s
    G = get_Gaussian(s_gauss, a=1.0*k)
    
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
def compute_PtsTensorMetrics(TensorMS, dfMS, scales, t0=0):
    start = time.time()- t0
 
    n = dfMS.shape[0]
    v_orientation = np.zeros((n,3))
    v_anisotropy = np.zeros(n)
    v_tubularity = np.zeros(n)
    v_disk = np.zeros(n)
    
    #Select Scales
    ss = np.unique(dfMS['S'].values)
    # print('')
    # print('ss')
    # print(ss)
    n_scales = ss.shape[0]
    k = 0
    for i in range(0, n_scales):
        s = ss[i]
        ix = np.where(scales==s)[0][0]
        TensorComponents = TensorMS[ix]
        
        #Select Points in the Scale       
        dfS = dfMS.loc[(dfMS['S'] == s)]  
        n_pts = dfS.shape[0] 
        # print()
        # print(dfS)
        dfS = dfS.astype(int)
        
        y, x, z = dfS['Y'].values, dfS['X'].values, dfS['Z'].values
        for j in range(0, n_pts):
            yxz = y[j], x[j], z[j]            
            pTensor = compute_pTensor(yxz, TensorComponents)    
            v_orientation[k,:], v_anisotropy[k], v_tubularity[k], v_disk[k] = compute_TensorMetrics(pTensor)
            
            #Compute Nearby Orientation Arround a Point
            # s = 0.5*s
            [v_pNear, v_v3Near, v_eigVal, v_eigVec] = compute_LocalEigen(TensorComponents, p0=yxz, R=s)
                    
            [m1, m2, m3] = [v_eigVal[:,0], v_eigVal[:,1], v_eigVal[:,2]]
            m = (m1 + m2 + m3)/3.0
            # print(m)
            
    
            #Average Tubularity
            v_tub =  (np.sqrt(m1**2 + m2**2))/np.abs(m3) - np.sqrt(2)
            avg_tub = v_tub.sum()/v_tub.shape[0]
            avg_tub = v_tub.mean()
            
            #Test
            # print('Test')
            # print('v3=', v_v3Near)
            #Angle Deviation
            v_avg = v_v3Near.sum(axis=0)
            v_avg = v_avg/np.sqrt((v_avg**2).sum())
            v1_u = v_avg
            v2_u = v_v3Near
            
            dot_prod = (v1_u*v2_u).sum(axis=1)
            dot_prod = np.abs(dot_prod)
            rho_rad = np.arccos(np.clip(dot_prod, -1.0, 1.0))
            rho_deg = rho_rad*(180/np.pi)
            rho_deg_std = rho_deg.std()
            # print(dot_prod)
            # print(dot_prod.shape)
        
            
            # v_tubularity[k] = avg_tub
            # v_tubularity[k] = np.sqrt((v_v3Near.sum(axis=0)**2).sum())
            # v_tubularity[k] = rho_deg_std
            
            k = k + 1            
            
    dfMS['Vx'] = v_orientation[:,0]
    dfMS['Vy'] = v_orientation[:,1]
    dfMS['Vz'] = v_orientation[:,2]
    
    dfMS['Ani'] = v_anisotropy
    dfMS['Tub'] = v_tubularity
    dfMS['Disk'] = v_disk
    
    stop = time.time()- t0
    return dfMS, start, stop
# =============================================================================
# 
# =============================================================================


    # v3= np.array([[-0.604409,    0.74529437, 0.28147123],
    #              [-0.51866984,  0.79953393,  0.30286479],
    #              [-0.49979517,  0.80551797,  0.31834822]])
    # mod = np.sqrt((v3.sum(axis=0)**2).sum())
# =============================================================================
# 
# =============================================================================
def compute_pTensor(yxz, TensorComponents):
    [x0, y0, z0] = yxz
    # [y0, x0, z0] = xyz
    [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = TensorComponents

    pTensor = np.asarray([[Dxx[x0,y0,z0], Dxy[x0,y0,z0], Dxz[x0,y0,z0]],
                          [Dxy[x0,y0,z0], Dyy[x0,y0,z0], Dyz[x0,y0,z0]],
                          [Dxz[x0,y0,z0], Dyz[x0,y0,z0], Dzz[x0,y0,z0]]])
    
    # myMax = np.abs(pTensor).max()
    # pTensor = pTensor/myMax
    # pTensor = np.round(pTensor, 3)
    
    return pTensor

# =============================================================================
# 
# =============================================================================
def compute_LocalEigen(TensorComponents, p0, R):
    
    #Matrix
    R = np.round(R).astype('int')
    r = np.linspace(-R, R, 2*R + 1)   
    v_pNear = np.array(list(itertools.product(r, repeat=3)))
    
    #Spherical Mask
    m = np.sqrt((v_pNear**2).sum(axis=1))
    mask = m<=R  
    v_pNear = v_pNear[mask]
    
    #Traslaion
    v_pNear[:, 0] = v_pNear[:, 0] + p0[0]
    v_pNear[:, 1] = v_pNear[:, 1] + p0[1]
    v_pNear[:, 2] = v_pNear[:, 2] + p0[2]
    v_pNear = v_pNear.astype('int')
    
    v_eigVal = []
    v_eigVec = []
    v_v3 = []
    n = v_pNear.shape[0]
    
    for i in range(0, n):
        pTensor = compute_pTensor(v_pNear[i,:], TensorComponents)
        eigVal, eigVec = compute_pEigen(pTensor) 
        v_eigVal.append(eigVal)
        v_eigVec.append(eigVec)  
        v_v3.append(eigVec[:,2])

    v_eigVal  = np.array(v_eigVal)
    v_eigVec  = np.array(v_eigVec)
    v_v3 = np.array(v_v3)
    
    return [v_pNear, v_v3, v_eigVal, v_eigVec]

#EigenValues & EigenVectors
def compute_pEigen(pTensor):
    
    eigVal, eigVec = np.linalg.eig(pTensor) 
  
    #Sort in Descending Order (v1>v2>v3)
    asc_ix = eigVal.argsort()[::-1]
    eigVal = eigVal[asc_ix]
    eigVec = eigVec[:, asc_ix] 
    
    return [eigVal, eigVec]
# =============================================================================
# 
# =============================================================================
def compute_TensorMetrics(pTensor):  
    
    #Eigenvalues
    eigVal, eigVec = compute_pEigen(pTensor) 
    
    m1 = eigVal[0]
    m2 = eigVal[1]
    m3 = eigVal[2]
    
    v1 = eigVec[:,0]
    v2 = eigVec[:,1]
    v3 = eigVec[:,2]

    #Normalization    
    k_norm = 1.0/np.sqrt(eigVal[0]**2+eigVal[1]**2+eigVal[2]**2)
    eigVal = k_norm*eigVal

    # Anisotropy (custom)
    eigValPairs = np.asarray([((eigVal[0]-eigVal[1])/(eigVal[0]+eigVal[1]))**2,
                              ((eigVal[0]-eigVal[2])/(eigVal[0]+eigVal[2]))**2,
                              ((eigVal[1]-eigVal[2])/(eigVal[1]+eigVal[2]))**2])         
    anisotropy = np.sqrt(np.sum(eigValPairs))
    
    # Anisotropy (custom)
    # anisotropy = (np.sqrt((m1**2 + m2**2)))/m3
    # anisotropy = m1/m2
    
    # ????? Anisotropy (FA: Factor of Anisotropy)
    # m = (m1 + m2 + m3)/3.0
    # FA = np.sqrt(2/3)*np.sqrt(((m1 - m)**2 + (m2 - m)**2 + (m3-m)**2)/((m1**2 + m2**2 + m3**2)))
    # anisotropy = FA

    #Tubularity
    tubularity =  (np.sqrt(m1**2 + m2**2))/np.abs(m3) - np.sqrt(2)
        
    #Disc
    disk = m1/(np.sqrt(m2**2 + m3**2)) - 1.0/np.sqrt(2)    
    
    #Orientation
    # ix = np.argmin(eigVal)
    # v = eigVec[:,ix]
    # orientation = v/(np.sqrt(v[0]**2 + v[1]**2 + v[2]**2))
    orientation = v3/(np.sqrt(v3[0]**2 + v3[1]**2 + v3[2]**2))
    
      
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
