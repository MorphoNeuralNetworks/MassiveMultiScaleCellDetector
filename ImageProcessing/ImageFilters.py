# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:46:02 2020

@author: pc
"""

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm


#from ImageProcessing.ImageMasks import get_Circle, get_Sphere, get_Cube
#from Convolution import convolve_3D_3x1D
   
#Tasks
#1) get_DoG_SpatialSpace
#2) get_DoG_FrequiencySpace
#3) get_DoG_1D
#4) get_DoG_2D
#5) get_DoG_3D
#6) Normalization Options: Gmax or Gopt
#7) Control DoG scale with r_cross instead of sc
#sc = 9.5
#rS = 1.5    #ratio of sigmas
#rV = 1.0    #ratio of volumes
#a  = 1.0    #constant
#
#r_cross = np.sqrt((a*sc**2)*(rS**2/(rS**2-1))*np.log(rS**2/rV))
#sc9 = np.sqrt(r_cross**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**2/rV))))
#print('')
#print(r_cross)
#print('')
#print(sc)
#print(sc9)

#Observation
#1) When the 1D-DoG filter is small the DC>0 even when rV=1.0
#   When sc<1.5 -> Vol=0.43 -> Vol shoud be Vol=0 -> DC is transmmited
#   When sc<6.0 -> Vol=0.10 -> Vol shoud be Vol=0 -> DC is transmmited
#   This might be solved by computing the filter in the Frequency Domain and applying the inverse fft

#==============================================================================
# Gaussian Function: 1D, 2D and 3D
#==============================================================================
def get_Gaussian(s, a=1.0):

    
    #Get Dimensions
    dim = len(s)

    #Discretization
    smax =np.max(s)
    Tr = int(np.ceil(3.0*smax))
    n = 1 + 2*Tr
    r = np.linspace(-Tr, +Tr, n)
    
    #1D-Gaussian
    if dim==1:
        #Unpaking the Parameters
        sx = s[0]
        
        #Discrete Samples
        xx = r
        
        #Gaussian (1D)   
        F= np.exp(-((xx**2)/(a*sx**2)))
        
        #Normalization 
        #k_norm = 1.0/(np.sum(np.abs(F)))
        k_norm = 1.0/(sx*(a*np.pi)**(1.0/2.0))
        F_norm = k_norm*F
    
    #2D-Gaussian
    elif dim==2:
        #Unpaking the Parameters 
        sx, sy = s
        
        #Discrete Samples
        [xx, yy] = np.meshgrid(r, r) 
        
        #Gaussian    
        F = np.exp(-((xx**2)/(a*sx**2) + (yy**2)/(a*sy**2) ))
        
        #Normalization 
        #k_norm = 1.0/(np.sum(np.abs(F)))
        k_norm = 1.0/((sx*(a*np.pi)**(1.0/2.0)) * (sy*(a*np.pi)**(1.0/2.0))) 
        F_norm = k_norm*F

    #3D-Gaussian
    elif dim==3:
        #Unpaking the Parameters 
        sx, sy, sz = s
        
        #Discrete Samples
        [xx, yy, zz] = np.meshgrid(r, r, r) 
        
        #Gaussian    
        F = np.exp(-((xx**2)/(a*sx**2) + (yy**2)/(a*sy**2) + (zz**2)/(a*sz**2) ))
        
        #Normalization 
        #k_norm = 1.0/(np.sum(np.abs(F)))
        k_norm = 1.0/((sx*(a*np.pi)**(1.0/2.0)) * (sy*(a*np.pi)**(1.0/2.0)) * (sz*(a*np.pi)**(1.0/2.0)) ) 
        F_norm = k_norm*F  
    else:
        print('Dimensions greater than 3 are not supported')
     
    return F_norm
    
    
#==============================================================================
# First Derivative of a Gaussian Function across x-axis: 1D, 2D and 3D
#==============================================================================

def get_DxGaussian(s, a=1.0):
    
    #Get Dimensions
    dim = len(s)

    #Discretization
    smax =np.max(s)
    Tr = int(2*np.ceil(3.0*smax))
    n = 1 + 2*Tr
    r = np.linspace(-Tr, +Tr, n)
    
    #1D-First Derivative of a Gaussian
    if dim==1:
        #Unpaking the Parameters
        sx = s[0]
        
        #Discrete Samples
        xx = r
        
        #First Derivative of a Gaussian   
        F = -xx*np.exp(-((xx**2)/(a*sx**2)))
        
        #Normalization
        #k_norm = 1.0/(np.sum(np.abs(F)))
        k_norm = ((np.sqrt(2)*np.exp(1.0/2.0))/(a*np.sqrt(np.pi)))**dim*(1.0/(sx**2))
        F_norm = k_norm*F


    #2D-First Derivative of a Gaussian
    elif dim==2:
        #Unpaking the Parameters
        sx, sy = s
        
        #Discrete Samples
        [xx, yy] = np.meshgrid(r, r) 
        
        #First Derivative of a Gaussian   
        F = -xx*np.exp(-( (xx**2)/(a*sx**2) + (yy**2)/(a*sy**2) ))
        
        #Normalization 
        #k_norm = 1.0/(np.sum(np.abs(F)))
        k_norm = (1.0/(sx*sy*(a*np.pi)**(1.0/2.0)))* (((np.sqrt(2)*np.exp(1.0/2.0))/(a*np.sqrt(np.pi))) * (1.0/(sx))) 
        F_norm = k_norm*F


    #3D-First Derivative of a Gaussian
    elif dim==3:
        #Unpaking the Parameters
        sx, sy, sz = s
        
        #Discrete Samples
        [xx, yy, zz] = np.meshgrid(r, r, r) 
        
        #First Derivative of a Gaussian   
        F = -xx*np.exp(-( (xx**2)/(a*sx**2) + (yy**2)/(a*sy**2) + (zz**2)/(a*sz**2) ))
        
        #Normalization
        #k_norm = 1.0/(np.sum(np.abs(F)))
        k_norm = (1.0/(sx*sy*sz*(a*np.pi)**(2.0/2.0)))*(((np.sqrt(2)*np.exp(1.0/2.0))/(a*np.sqrt(np.pi))) * (1.0/(sx)))
        F_norm = k_norm*F 

    else:
        print('Dimensions greater than 3 are not supported')
    
    return F_norm        
#Note: Normalization Components
#k_norm = (2.0/a) * (1.0/np.sqrt(a*np.pi*sx**6)) * (1.0/((np.sqrt(2)/(np.sqrt(a)*sx))*np.exp(-1.0/2.0)))

#==============================================================================
#Magnitude of the Second Derivative of a Gaussian
#Module DoG Function: 1D, 2D and 3D
#==============================================================================
#Note that the DoG function aproximates the trace of a Second Derivative of a Gaussian
#DoG = np.sqrt(DDx**2 + DDy**2 )

def get_DoG(s, rS=1.1, rV=1.0, a=1.0):
    
    #Get Dimensions
    dim = len(s)

    
    #1D-DoG
    if dim==1:
        #Unpaking the Parameters
        rx = s[0]
        
        #Variable Change: control the Spatial Scale of the DoG with r_cross
        sx = np.sqrt(rx**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS/rV))))
        s = [sx]
        
        #Discretization
        smax =np.max(s)
        Tr = int(np.ceil(3.0*smax*rS))
        n = 1 + 2*Tr
        r = np.linspace(-Tr, +Tr, n)
        xx = r
    
        #Difference of a Gaussian (DoG)
        Gc = rS*np.exp(-(xx**2)/(a*sx**2))
        Gs = rV*np.exp(-(xx**2)/(a*sx**2*rS**2))
        
        #Normalization
        A = (a*np.pi*sx**2*rS**2)**(1.0/2.0)
        B = ((1.0/rV)*(1.0/rS**2))**(1.0/(rS**2 - 1)) - rV*((1.0/rV)*(1.0/rS**2))**(rS**2/(rS**2 - 1))
        K_norm = 1.0/(A*B)
        DoG = K_norm*(Gc-Gs)
 
    #2D-DoG
    elif dim==2:
        #Unpaking the Parameters
        rx, ry = s
        
        #Variable Change: control the Spatial Scale of the DoG with r_cross
        sx = np.sqrt(rx**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**2/rV))))
        sy = np.sqrt(ry**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**2/rV))))
        s = [sx, sy]
        
        #Discretization
        smax =np.max(s)
        Tr = int(np.ceil(3.0*smax*rS))
        n = 1 + 2*Tr
        r = np.linspace(-Tr, +Tr, n)
        [xx, yy] = np.meshgrid(r, r) 
    
        #Difference of a Gaussian (DoG)
        Gc = rS**2*np.exp(-( (xx**2)/(a*sx**2)       + (yy**2)/(a*sy**2) ) )
        Gs = rV*np.exp(-( (xx**2)/(a*sx**2*rS**2) + (yy**2)/(a*sy**2*rS**2) ) )
        
        #Normalization
        A = ((a*np.pi*sx**2*rS**2)**(1.0/2.0)) * ((a*np.pi*sy**2*rS**2)**(1.0/2.0))
        B = ((1.0/rV)*(1.0/rS**2))**(1.0/(rS**2 - 1)) - rV*((1.0/rV)*(1.0/rS**2))**(rS**2/(rS**2 - 1))
        K_norm = 1.0/(A*B)
        print()
        DoG = K_norm*(Gc-Gs)
        
    #3D-DoG
    elif dim==3:
        #Unpaking the Parameters
        rx, ry, rz = s
        
        #Variable Change: control the Spatial Scale of the DoG with r_cross
        sx = np.sqrt(rx**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**3/rV))))
        sy = np.sqrt(ry**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**3/rV))))
        sz = np.sqrt(rz**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**3/rV))))
        s = [sx, sy, sz]
        
        #Discretization
        smax =np.max(s)
        Tr = int(np.ceil(3.0*smax*rS))
        n = 1 + 2*Tr
        r = np.linspace(-Tr, +Tr, n)
        [xx, yy, zz] = np.meshgrid(r, r, r) 
    
        #Difference of a Gaussian (DoG)
        Gc = rS**3*np.exp(-( (xx**2)/(a*sx**2) + (yy**2)/(a*sy**2) + (zz**2)/(a*sz**2)) )
        Gs = rV*np.exp(-( (xx**2)/(a*sx**2*rS**2) + (yy**2)/(a*sy**2*rS**2) + (zz**2)/(a*sz**2*rS**2) ) )
        
        #Normalization
        A = ((a*np.pi*sx**2*rS**2)**(1.0/2.0)) * ((a*np.pi*sy**2*rS**2)**(1.0/2.0)) * ((a*np.pi*sz**2*rS**2)**(1.0/2.0))
        B = ((1.0/rV)*(1.0/rS**2))**(1.0/(rS**2 - 1)) - rV*((1.0/rV)*(1.0/rS**2))**(rS**2/(rS**2 - 1))
        K_norm = 1.0/(A*B)
        DoG = K_norm*(Gc-Gs)
        
    return DoG


#==============================================================================
#     Visualization of the above Filters
#==============================================================================

def plot_1DFilter(Fx):
    
    #Plot Settings
    ny, nx = 1, 2
    m = 0.75
    fig, axs = plt.subplots(ny,nx)
    graphSize = [6.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
    fig.set_size_inches(graphSize) 
    
    #Spatial Space
    nx = Fx.shape [0]   
    plot_Filter(axs[0], Fx)
    
    #Frequency Space
    n = nx
    if n<101:
        Nf = 201
    else:
        Nf = get_Odd(n)
        
    Fx_fft = np.fft.fftshift(np.fft.fft(Fx,(Nf))) 
    Fx_fft = np.abs(Fx_fft) 
 
    plot_Filter_fft(axs[1], Fx_fft)
    
    
    print('DC', Fx.sum())
    print('G_Max', Fx_fft.max())
    
    return fig, axs
    

def plot_2DFilter(Fxy):
    
    #Plot Settings
    ny, nx = 2, 2
    m = 0.75
    fig, axs = plt.subplots(ny,nx)
    graphSize = [6.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
    fig.set_size_inches(graphSize) 
    
    #Spatial Space
    [ny, nx] = Fxy.shape
    Fx = Fxy[ny//2, :    ]
    Fy = Fxy[:    , nx//2]
    
    plot_Filter(axs[0,0], Fx)
    plot_Filter(axs[1,0], Fy)
    
    #Frequency Space
    n = np.max([ny, nx])
    if n<101:
        Nf = 101
    else:
        Nf = get_Odd(n)
    Fxy_fft = np.fft.fftshift(np.fft.fft2(Fxy,(Nf,Nf))) 
    Fxy_fft = np.abs(Fxy_fft) 
    
    Fx_fft = Fxy_fft[Nf//2, :    ]
    Fy_fft = Fxy_fft[:    , Nf//2]
 
    plot_Filter_fft(axs[0,1], Fx_fft)
    plot_Filter_fft(axs[1,1], Fy_fft)
    
    
    print('DC', Fxy.sum())
    print('G_Max', Fxy_fft.max())
    
    return fig, axs



def plot_3DFilter(Fxyz):
    
    #Plot Settings
    ny, nx = 3, 2
    m = 0.75
    fig, axs = plt.subplots(ny,nx)
    graphSize = [6.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
    fig.set_size_inches(graphSize) 
    
    #Spatial Space
    [ny, nx, nz] = Fxyz.shape
    Fx = Fxyz[ny//2, :    , nz//2]
    Fy = Fxyz[:    , nx//2, nz//2]
    Fz = Fxyz[ny//2, ny//2, :    ]
    
    plot_Filter(axs[0,0], Fx)
    plot_Filter(axs[1,0], Fy)
    plot_Filter(axs[2,0], Fz)
    
    #Frequency Space
    n = np.max([ny, nx, nz])
    if n<101:
        Nf = 101
    else:
        Nf = get_Odd(n)
    Fxyz_fft = np.fft.fftshift(np.fft.fftn(Fxyz,(Nf,Nf,Nf))) 
    Fxyz_fft = np.abs(Fxyz_fft) 
    
    Fx_fft = Fxyz_fft[Nf//2, :    , Nf//2]
    Fy_fft = Fxyz_fft[:    , Nf//2, Nf//2]
    Fz_fft = Fxyz_fft[Nf//2, Nf//2, :    ]
 
    plot_Filter_fft(axs[0,1], Fx_fft)
    plot_Filter_fft(axs[1,1], Fy_fft)
    plot_Filter_fft(axs[2,1], Fz_fft)   
    
    
    print('DC', Fxyz.sum())
    print('G_Max', Fxyz_fft.max())
    
    return fig, axs

#==============================================================================
#     
#==============================================================================

def plot_Filter(ax, F):    
    nx = F.shape[0] 
    nx2 = nx//2    
    x = np.linspace(-nx2, +nx2, nx) 

    ax.plot(x, F, marker='o')
    ax.hlines(y=0, xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='k')
    ax.vlines(x=0, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')


def plot_Filter_fft(ax, F_fft):
    Nf = F_fft.shape[0]    
    F_fft = F_fft[Nf//2:]
    f = np.linspace(0, 0.5, Nf//2 + 1)

    ax.plot(f, F_fft)
    ax.set_ylim([0, ax.axes.get_ylim()[1]])
    ax.set_xlim([0, 0.5])


def get_Odd(num):
    num = int(num)
    if (num % 2) == 0: 
        num = num + 1
    return num

#==============================================================================
#     
#==============================================================================

def doggen(sigma):
      """
      Helper function to generate derivatives of Gaussian kernels, in either 1D, 2D, or 3D.
      Source code in MATLAB obtained from Qiyuan Tian, Stanford University, September 2015
      :param sigma: Sigma for use (see defaults in generate_FSL_structure_tensor)
      :return: Derivative of Gaussian kernel with dimensions of sigma.
      """
      halfsize =2*np.ceil(3 * np.max(sigma))
      x = range(np.single(-halfsize), np.single(halfsize + 1));  # Python colon is not inclusive at end, while MATLAB is.
      dim = len(sigma);

      if dim == 1:
          X = np.array(x);  # Remember that, by default, numpy arrays are elementwise multiplicative
          X = X.astype(float);
          k = -X * np.exp(-X**2/(2 * sigma**2));

      elif dim == 2:
          [X, Y] = np.meshgrid(x, x);
          X = X.astype(float);
          Y = Y.astype(float);
          k = -X * np.exp(-X**2/(2*sigma[0]**2) + np.exp(-Y**2))
          k = -X*np.exp(np.divide(-np.power(X, 2), 2 * np.power(sigma[0], 2))) * np.exp(np.divide(-np.power(Y,2), 2 * np.power(sigma[1],2)))


      elif dim == 3:
          [X, Y, Z] = np.meshgrid(x, x, x);
          X = X.transpose(0, 2, 1);  # Obtained through vigorous testing (see below...)
          Y = Y.transpose(2, 0, 1);
          Z = Z.transpose(2, 1, 0);

          X = X.astype(float);
          Y = Y.astype(float);
          Z = Z.astype(float);
          k = -X * np.exp(np.divide(-np.power(X, 2), 2 * np.power(sigma[0], 2))) * np.exp(np.divide(-np.power(Y,2), 2 * np.power(sigma[1],2))) * np.exp(np.divide(-np.power(Z,2), 2 * np.power(sigma[2],2)))

      else:
          print 'Only supports up to 3 dimensions'

      
      k_norm = (np.sum(np.abs(k[:])))
#      print(k_norm)
#      k_norm = 2*np.pi*sigma[0]**4
#      print(k_norm)
#      k_norm =  (2*np.pi)**(1.0)*sigma[0]**3*np.exp(-1.0/2.0)
      k = np.divide(k, k_norm);
      return k 
if __name__== '__main__':
    

    
#==============================================================================
#     
#==============================================================================
    a = 1.0
    
    #1D-Sigmas
    s = [2.0]
    s = [7.0]
    
    #2D-Sigmas
    s = [1.0, 1.0]
    s = [2.0, 2.0]
    s = [3.0, 3.0]
#    s = [7.0, 7.0]
#    s = [7.0, 2.0]
#    s = [2.0, 7.0]
    s = [12.0, 12.0]
#     
#    #3D-Sigmas
#    s = [2.0, 2.0, 2.0]
#    s = [3.0, 3.0, 3.0]
#    s = [7.0, 7.0, 7.0]
#    s = [7.0, 7.0, 2.0]
#    s = [2.0, 2.0, 7.0]
#    s = [2.0, 5.0, 7.0]
    

    F = get_Gaussian(s, a=1.0)
    F = get_DxGaussian(s, a=0.25)    
#    F = get_DxGaussian(s, a=2.0) 
#    F = doggen(s)
    F = get_DoG(s, rS=1.1, a=1.0)
#    F = get_DoG(s, rS=1.1, rV=0.96, a=1.0)
#    F = get_DoG(s, rS=3.5, a=1.0)
    
    #Transpose
#    F = np.transpose(F, (1, 0, 2))
#    F = np.transpose(F, (0, 2, 1))
    dim = len(s)
    
#==============================================================================
#   
#==============================================================================
    if dim==1:
        Fx = F    
        fig, axs = plot_1DFilter(Fx)
        fig.tight_layout(h_pad=1.0)  
        plt.show()


    if dim==2:
        Fxy = F  
        plt.imshow(Fxy, cmap=cm.Greys_r,  interpolation='nearest')
        plt.show()
        fig, axs = plot_2DFilter(Fxy)
        fr= 1.0/(4*s[0])
        ax = axs[0,1]
        ax.vlines(x=fr, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r')
        fig.tight_layout(h_pad=1.0)  
        plt.show()


        
    if dim==3:
        Fxyz = F 
        ny, nx, nz = Fxyz.shape
        plt.imshow(Fxyz[:,:,nz//2], cmap=cm.Greys_r,  interpolation='nearest')
        plt.show()
        fig, axs = plot_3DFilter(Fxyz)
        fig.tight_layout(h_pad=1.0)  
        plt.show()
    
#    print('')
#    print( (np.sqrt(2)/(np.sqrt(a)*sx))*np.exp(-1.0/2.0) )
#    print( (a/2.0)*(np.sqrt(2)/(np.sqrt(a)*sx))*np.exp(-1.0/2.0) )

    
#    f = np.linspace(0,0.5,100)
#    y_fft = 2*np.pi*f*np.exp(-a*np.pi**2*sx**2*f**2)
##    y = 
##    Fx_fft = np.fft.fftshift(np.fft.fft(Fx,(Nf))) 
#    plt.plot(f,y_fft)
#    plt.show()
#    print(y.max())
    
#    f = np.linspace(0,0.5,100)
#    y = a*np.pi*f*np.exp(-a*np.pi**2*sx**2*f**2)
#    axs[1].plot(f,y)
#==============================================================================
#     
#==============================================================================
    


#    m = 0.4
#    w, h = 2, 1
#    graph_size = (m*w*10.2, m*h*10.2)
#==============================================================================
#     Input
#==============================================================================   
#    s = 3    #sigma center
#    rS = 2.5    #ratio of sigmas
#    rV = 1.0    #ratio of volumes
#    a  = 1.0    #constant
#==============================================================================
#     
#==============================================================================
#    fc = np.sqrt((1.0/(a*np.pi**2*sc**2))*(1.0/(1-rS**2))*np.log(1.0/(rS**2*rV)))
#    r_cross = 1.0*sc
#    sc_aux = np.sqrt(r_cross**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS/rV))))    

#    fc = 0.25*(1.0/sc)
#    scAux = np.sqrt((1.0/(a*np.pi**2*fc**2))*(1.0/(1-rS**2))*np.log(1.0/(rS**2*rV)))
#    r_cross = np.sqrt((a*scAux**2)*(rS**2/(rS**2-1))*np.log(rS/rV))
#    
#    r_cross = np.sqrt((a*s**2)*(rS**2/(rS**2-1))*np.log(rS/rV))
#    ax = axs[0,0]
#    ax.vlines(x=s, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#    n = Fxyz.shape[0]
#    img = Fxyz[:,:,n//2]
#    plt.imshow(img, cmap=cm.Greys_r,  interpolation='nearest')

#==============================================================================
#   1D
#==============================================================================
#    Fx = get_DoG1D(sx=s, rS=rS, rV=rV)       
#    fig, axs = plot_1DFilter(Fx)
#    fig.tight_layout(h_pad=1.0)  
#    plt.show()
#    
##==============================================================================
##   2D
##==============================================================================
#    s = 11.0    
#    Fxy = get_DoG2D(sx=s, sy=s, rS=rS, rV=rV)     
#    
#    fig, axs = plot_2DFilter(Fxy)
#    fig.tight_layout(h_pad=1.0)  
#    plt.show()
    
#==============================================================================
#     3D
#==============================================================================
#    Fxyz = get_DoG3D(sx=s, sy=s, sz=s, rS=rS, rV=rV)     
#    fig, axs = plot_3DFilter(Fxyz)
#    fig.tight_layout(h_pad=1.0)  
#    plt.show()
    

 
#==============================================================================
#   3D Equivalent Separate Filter  (Fx, Fy, Fz)
#==============================================================================
#    Fx = get_DoG1D(sx=s, rS=rS, rV=rV) 
#    n = Fx.shape[0]
#    imgIn = np.zeros((n,n,n))
#    imgIn[n//2,n//2,n//2] = n
#    Fxyz = convolve_3D_3x1D(imgIn, Fx, Fx, Fx) 
#    fig, axs = plot_3DFilter(Fxyz)
#    fig.tight_layout(h_pad=1.0)  
#    plt.show()


#==============================================================================
#    3D - Iso Function
#==============================================================================
#    Fxyz = get_DoG3D_Iso(sc=s, rS=rS, rV=rV)     
#    fig, axs = plot_3DFilter(Fxyz)
#    fig.tight_layout(h_pad=1.0)  
#    plt.show()

#==============================================================================
#     3D Gauss
#==============================================================================
#    Fxyz = get_Gauss3D(sx=s, sy=s, sz=s, rS=2.5, rV=rV)     
#    fig, axs = plot_3DFilter(Fxyz)
#    fig.tight_layout(h_pad=1.0)  
#    plt.show()
    
#    rS=1.0
#    np.sqrt(s**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**3/rV))))

#==============================================================================
# 
#==============================================================================

#    r_cross = sc
#    sc = np.sqrt(r_cross**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**2/rV))))   
#    fc = np.sqrt((1.0/(a*np.pi**2*sc**2))*(1.0/(1-rS**2))*np.log(1.0/(rS**2*rV))) 














#==============================================================================
#   1D     
#==============================================================================
#    Fx = get_DoG1D(sc=sc, rS=rS, rV=rV)
#    
#    fc = 0.25*(1.0/sc)
#    scAux = np.sqrt((1.0/(a*np.pi**2*fc**2))*(1.0/(1-rS**2))*np.log(1.0/(rS**2*rV)))
#    r_cross = np.sqrt((a*scAux**2)*(rS**2/(rS**2-1))*np.log(rS/rV))
#
#    #Ploting
#    fig, axs = plt.subplots(1,2) 
#    fig.set_size_inches(graph_size)
#    plot_Filter(Fx, axs)
#    
#    ax = axs[0]
#    ax.vlines(x=r_cross, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
##
#    ax = axs[1]
#    ax.vlines(x=(4*r_cross)**-1, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r')
#    ax.vlines(x=fc, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='-', color='k')
#    plt.show()
#    
#    s_th = 0.25/fc
#    s_ex = r_cross
#    print('s_rc=', s_ex)
#    print('s_fc=', s_th)
#    print(s_ex/s_th)


    
#==============================================================================
#     3D equivalent separate fitlers (Fx + Fy + Fz)
#==============================================================================
        
#    Fx = get_DoG1D(sc=sc, rS=rS, rV=rV) 
#    n = Fx.shape[0]
#    imgIn = np.zeros((n,n,n))
#    imgIn[n//2,n//2,n//2] = n
#    Fxyz = convolve_3D_3x1D(imgIn, Fx, Fx, Fx)  
#    
#    #Ploting
#    fig, axs = plt.subplots(1,2) 
#    fig.set_size_inches(graph_size)
#    plot_Filter(Fxyz, axs)
#    
#    ax = axs[0]
#    ax.vlines(x=r_cross, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#
#    ax = axs[1]
#    ax.vlines(x=(4*r_cross)**-1, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r')
#    ax.vlines(x=fc, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='-', color='k')
#
#
#    plt.show()
#    print ('DoG 3D, DC=', Fxyz.sum())
#    
#==============================================================================
#     3D equivalent separate fitlers (Fxy + Fz)
#==============================================================================
#    import cv2 
#    Fxy = get_DoG2D(sc=sc, rS=rS, rV=rV) 
#    Fz  = get_DoG1D(sc=sc, rS=rS, rV=rV) 
#    n = Fxy.shape[0]
#    imgIn = np.zeros((n,n,n))
#    imgIn[n//2,n//2,n//2] = n
#    
#    
#    Fxyz = cv2.filter2D(imgIn, -1, Fxy, borderType=0) 
#    Fxyz = cv2.filter2D(np.transpose(Fxyz, axes=(2,1,0)), -1, Fz, borderType=0)
#    Fxyz = np.transpose(Fxyz, axes=(2,1,0))
#     
##    Fxy = Fxyz
#    #Ploting
#    fig, axs = plt.subplots(1,2) 
#    fig.set_size_inches(graph_size)
#    plot_Filter(Fxyz, axs)
#    
#    ax = axs[0]
#    ax.vlines(x=r_cross, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#
#    ax = axs[1]
#    ax.vlines(x=(4*r_cross)**-1, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r')
#    ax.vlines(x=fc, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='-', color='k')
#
#
#    plt.show()
#    print ('DoG 3D, DC=', Fxyz.sum()) 



#    
#    print ('DoG 1D, DC=', Fx.sum())
#    print ('DoG 1D, Gmax=', H_fft_M.max())
    
#==============================================================================
#    2D 
#==============================================================================

#    Fxy = get_DoG2D(sc=sc, rS=rS, rV=rV) 
    
#    r_aux = 1.0*sc
#    r_cross = np.sqrt((a*sc**2)*(rS**2/(rS**2-1))*np.log(rS**2/rV))
#    r_cross = sc
#    sc = np.sqrt(r_cross**2*((rS**2 - 1)/(rS**2))*(1.0/(a*np.log(rS**2/rV))))   
#    fc = np.sqrt((1.0/(a*np.pi**2*sc**2))*(1.0/(1-rS**2))*np.log(1.0/(rS**2*rV))) 
#    
#    #Ploting
#    fig, axs = plt.subplots(1,2) 
#    fig.set_size_inches(graph_size)
#    plot_Filter(Fxy, axs)
#    
#    ax = axs[0]
#    ax.vlines(x=r_cross, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#
#    ax = axs[1]
#    ax.vlines(x=(4*r_cross)**-1, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r')
#    ax.vlines(x=fc, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='-', color='k')
#
#    s_th = 1.0/fc
#    s_ex = 2*r_aux
#    print(s_ex)
#    print(s_th)
#    print(s_ex/s_th)
#    plt.show()
#    
#    
#    print ('DoG 2D, DC=', Fxy.sum())
#    print ('DoG 2D, Gmax=', H_fft_M.max())
 
#==============================================================================
#   3D
#==============================================================================
#    Fxyz = get_DoG3D(sc=sc, rS=rS, rV=rV) 
#    
#    #Ploting
#    fig, axs = plt.subplots(1,2) 
#    fig.set_size_inches(graph_size)
#    plot_Filter(Fxyz, axs)
#    
#    ax = axs[0]
#    ax.vlines(x=r_cross, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#
#    ax = axs[1]
#    ax.vlines(x=(4*r_cross)**-1, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r')
#    ax.vlines(x=fc, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='-', color='k')
#
#
#    plt.show()
#    
#    print ('DoG 3D, DC=', Fxyz.sum())
#    print ('DoG 3D, Gmax=', H_fft_M.max())
    
       

#==============================================================================
#   Draft  
#==============================================================================
#    n = F.shape[0]
#    n_half = int((n-1)/2)
#    x = np.linspace(-n_half, +n_half, n)
#    
#    nDim = len(F.shape)
#    if nDim==1:
#        y = F
#    elif nDim==2:
#        y = F[n_half, :]
#    elif nDim==3:
#        y = F[n_half, n_half, :]
#    
##    F[F<np.min(y)] = 0
##    print('DC=', F.sum())
#
#    
#    fig, axs = plt.subplots(1,1)
#    ax = axs    
#    ax.plot(x,y, marker='o')
#    ax.hlines(y=0, xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='k')
#    ax.vlines(x=0, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#    plt.show()
    
      


    
    
#==============================================================================
#     
#==============================================================================
#    R = 5.0
#    imgIn = get_Circle(r1=R, r2=2*R)
#    
#    R=5
#    #Separate Filter
#    Fx = get_DoG1D(sc=R, rS=rS, rV=rV) 
#    imgOut1 = convolve_2D(imgIn, Fx, Fx)
#    print(imgOut1.max())
#    
#    #Separate Filter
#    Fxy = get_DoG2D(sc=R, rS=rS, rV=rV) 
#    import cv2
#    imgOut2 = cv2.filter2D(imgIn, -1, Fxy) 
#    print(imgOut2.max())

    

#==============================================================================
# Draft: Normalization of First Derivative of a Gaussian
#==============================================================================
#    
#    a = 2.0
#    s = [5.0]
##    s = [5.0, 2.0]
##    s = [1.0, 7.0, 3.0]
#    #------------
#    
#    dim = len(s)
##==============================================================================
##     
##==============================================================================
##    G = get_Gaussian(s)
#    #Get Dimensions
#    dim = len(s)
#
#    #Discretization
#    smax =np.max(s)
#    Tr = int(np.ceil(3.0*smax))
#    n = 1 + 2*Tr
#    r = np.linspace(-Tr, +Tr, n)
#    
#    #1D-Gaussian
#    if dim==1:
#        #Unpaking the Parameters
#        sx = s[0]
#        
#        #Discrete Samples
#        xx = r
#        
#        #Gaussian (1D)   
#        F = -xx*np.exp(-((xx**2)/(a*sx**2)))
#        
#        #Normalization 
#        k_norm = (2.0/a) * (1.0/np.sqrt(a*np.pi*sx**6)) * (1.0/((np.sqrt(2)/(np.sqrt(a)*sx))*np.exp(-1.0/2.0)))
#        k_norm = ((np.sqrt(2)*np.exp(1.0/2.0))/(a*np.sqrt(np.pi)))*(1.0/sx**2)
#
##        k_norm = 1.0
#        F_norm = k_norm*F
#
##==============================================================================
##   
##==============================================================================
#    if dim==1:
#        Fx = F_norm    
#        fig, axs = plot_1DFilter(Fx)
#        fig.tight_layout(h_pad=1.0)  
#        
#        f = np.linspace(0,0.5,100)
#        y = a*np.pi*f*np.exp(-a*np.pi**2*sx**2*f**2)
#        axs[1].plot(f,y)
#        plt.show()
#
#
#    if dim==2:
#        Fxy = F_norm   
#        fig, axs = plot_2DFilter(Fxy)
#        fig.tight_layout(h_pad=1.0)  
#        plt.show()
#        
#    if dim==3:
#        Fxyz = F_norm    
#        fig, axs = plot_3DFilter(Fxyz)
#        fig.tight_layout(h_pad=1.0)  
#        plt.show()
#    
#    print('')
#    print( (np.sqrt(2)/(np.sqrt(a)*sx))*np.exp(-1.0/2.0) )
#    print( (a/2.0)*(np.sqrt(2)/(np.sqrt(a)*sx))*np.exp(-1.0/2.0) )







