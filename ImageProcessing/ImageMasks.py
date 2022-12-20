# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:12:03 2020

@author: pc
"""


import numpy as np


from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def get_Square(n, I=+1.0):
    img2D = I*np.ones((n,n))
    return img2D

def get_Cube(n, I=+1.0):
    img3D = I*np.ones((n,n,n))
    return img3D
    

def get_On(r1, r2, Imin=0.0, Imax=+1.0):  
    r1 = get_Odd(r1)
    r2 = get_Odd(r2)
    
    r_half = r2//2   
    r = np.linspace(-r_half, r_half, r2)
    
    rho = np.sqrt(r**2)
    rho = np.round(rho)
    
    r_half = r1//2 
    rho[rho<r_half] = Imax
    rho[rho>=r_half] = Imin   
    
    return rho
    
def get_Circle(r1, r2, Imin=-1.0, Imax=+1.0):  
    r1 = np.round(r1)
    r2 = np.round(r2)  
     
    r = np.linspace(-r2, r2, 2*r2 + 1)    
    [XX, YY] = np.meshgrid(r,r)
    rho = np.sqrt(XX**2 + YY**2)
    rho = np.round(rho)   
    
    r_half = r1
    rho[rho<=r_half] = Imax
    rho[rho>r_half] = Imin   
    
    return rho

def get_Ellipsoid(R, n=None,  A=[0,0,0], T=[0,0,0], Imin=-1.0, Imax=+1.0):  
    Rx, Ry, Rz = R
    a, b, c = A
    # az, ay, ax = a*np.pi/180.0, b*np.pi/180.0, c*np.pi/180.0
    ax, ay, az = a*np.pi/180.0, b*np.pi/180.0, c*np.pi/180.0
    Rx, Ry, Rz = int(Rx), int(Ry), int(Rz)
    X, Y, Z = T
            
    Rmax = np.max([Rx, Ry, Rz])
    if (n<(2*Rmax+1)) or (not n):
        n = 2*Rmax+1
    else:        
        n = get_Odd(n)
        
    nx2, ny2, nz2 = n//2, n//2, n//2
  
    
    y, x, z = np.ogrid[-nx2:+nx2+1, -ny2:+ny2+1, -nz2:+nz2+1]
   
   
#    xx = x*np.cos(az)+y*np.sin(az)
#    yy = x*np.sin(az)-y*np.cos(az)
#    zz = z
    
    xx = (  x*(np.cos(ay)*np.cos(az)) +
            y*(-np.cos(ax)*np.sin(az) + np.sin(ax)*np.sin(ay)*np.cos(az)) +
            z*(np.sin(ax)*np.sin(az) +  np.cos(ax)*np.sin(ay)*np.cos(az))
            +X
            )
    
    yy = (  x*(np.cos(ay)*np.sin(az)) +
            y*(np.cos(ax)*np.cos(az) + np.sin(ax)*np.sin(ay)*np.sin(az)) +
            z*(-np.sin(ax)*np.cos(az) +  np.cos(ax)*np.sin(ay)*np.sin(az))
            +Y
            )
            
    zz = (  x*(-np.sin(ay)) +
            y*(np.sin(ax)*np.cos(ay)) +
            z*(np.cos(ax)*np.cos(ay))
            +Z
            )
    mask = (xx**2/float(Rx**2) + yy**2/float(Ry**2) + zz**2/float(Rz**2)) <= 1.0

    img = Imin*np.ones((n, n, n))
    img[mask] = Imax
    
    return img
    

#def get_Ellipsoid(Rx, Ry, Rz, n, Imin=-1.0, Imax=+1.0):  
#    
#    Rx, Ry, Rz, n = int(Rx), int(Ry), int(Rz), int(n)
#    n = get_Odd(n)
#    Rmax = np.max([Rx, Ry, Rz])
#    if n<(2*Rmax+1):
#        n = 2*Rmax+1
#    nx2, ny2, nz2 = n//2, n//2, n//2
#  
#    
#    y, x, z = np.ogrid[-nx2:+nx2+1, -ny2:+ny2+1, -nz2:+nz2+1]
#    mask = (x**2/float(Rx**2) + y**2/float(Ry**2) + z**2/float(Rz**2)) <= 1.0
#    
#    img = Imin*np.ones((n, n, n))
#    img[mask] = Imax
#    
#    return img
    
    
def get_Sphere(r1, r2, Imin=-1.0, Imax=+1.0):  
    r1 = np.round(r1)
    r2 = np.round(r2) 
    
    r = np.linspace(-r2, r2, 2*r2 + 1)  
    
    [XX, YY, ZZ] = np.meshgrid(r, r, r)
    rho = np.sqrt(XX**2 + YY**2 + ZZ**2)
#    rho = np.round(rho)
#    rho = np.ceil(rho)
#    rho = rho/np.sqrt(2)
    
    r_half = r1 
    rho[rho<=r_half] = Imax
    rho[rho>r_half] = Imin
    
    return rho    


def get_Sine2D(A = 1.0, OffSet=0.0, T=25,  N=200):  
    if N<T:
       N = T            
    n_cycles = N/float(T)
    
    x = np.linspace(0, n_cycles*2*np.pi, N)        
    y = OffSet - A*np.sin(x)          
    img = y*np.ones((x.shape[0],x.shape[0])) 
    
    return img
    
    
def get_Sine3D(A = 1.0, OffSet=0.0, T=25,  N=100):  
#    if N<T:
#       N = T            
#    n_cycles = N/float(T)
#    
#    r = np.linspace(0, n_cycles*2*np.pi, N)        
#    r = OffSet - A*np.sin(r)          
##    img = r*np.ones((r.shape[0],r.shape[0])) 
#    
#    r = np.linspace(-(N//2), N//2, 2*N + 1) 
#    [XX, YY, ZZ] = np.meshgrid(r, r, r)
#    rho = np.sqrt(XX**2 + YY**2 + ZZ**2)
#    rho = OffSet - A*np.cos(rho)
#    rho = np.round(rho)
    
    if N<T:
       N = T            
    n_cycles = N/float(T)   
    
     
    N = n_cycles*T
    N = (N+1)/2
    x1 = np.linspace(n_cycles*np.pi,0, N)
    x2 = np.linspace(0,n_cycles*np.pi, N)
    x2 = np.delete(x2,0)
    x = np.concatenate((x1,x2))
    [XX, YY, ZZ] = np.meshgrid(x, x, x)
    rho = np.sqrt(XX**2 + YY**2 + ZZ**2)
    
    img = OffSet + A*np.cos(rho) 
    return img
#==============================================================================
# 
#==============================================================================
def get_Odd(num):
    num = int(num)
    if (num % 2) == 0: 
        num = num + 1
    return num


 
  
if __name__== '__main__':
 

    Rx, Ry, Rz = 30, 5, 5
    k = 1.5
    n = np.round(2*k*np.max([Rx, Ry, Rz])).astype(int)
    Imin=-1.0
    Imax=+1.0
    imgIn = get_Ellipsoid(R=[Rx, Ry, Rz], n=n, A=[0,0,45], Imin=-1.0, Imax=+1.0)
    
    #Noise Model
    ny, nx, nz = imgIn.shape
    Noise = 2*np.random.rand(ny, nx, nz) - 1
    imgIn = imgIn + Noise

    
    ny, nx, nz = imgIn.shape
    imgIn = imgIn[:,:,nz//2]
    fig, axs = plt.subplots(1,1)
    ax = axs
    ax.imshow(imgIn,  cm.Greys_r, interpolation='nearest') 
    plt.show()
   
#==============================================================================
#     Circle
#==============================================================================
#    #Create a Circle: R, r
#    r1, r2 = 5, 5
#    r1, r2 = 5, 5
#    r1, r2 = 9, 9
#    r1, r2 = 15, 15
#    C = get_Circle(r1, r2)
#    
#    fig, axs = plt.subplots(1,1)
#    ax = axs
#    ax.imshow(C,  cm.Greys_r, interpolation='nearest') 
#    plt.show()
    
#==============================================================================
#   Sphere
#==============================================================================
#    #Create a Sphere: R, r
##    r1, r2 = 10, 20
#    S = get_Sphere(r1, r2, Imin=0.0, Imax=+1.0)
#    
#    img3D = S
#    ny, nx, nz = img3D.shape 
#    ny2, nx2, nz2 = ny//2, nx//2, nz//2
#    x = np.linspace(-nx2, +nx2, nx)
#    y = np.linspace(-ny2, +ny2, ny)
#    z = np.linspace(-nz2, +nz2, ny)
#    
#   
#    fig, axs = plt.subplots(1,1)
#    ax = axs
#    ax.imshow(img3D[:,:,nz2],  cm.Greys_r, interpolation='nearest') 
##    ax.grid(color='g', linestyle='-', linewidth=2)
#    plt.show()
#    print('N=',img3D.shape)
#    V_the = 4.0/3.0*(np.pi*r1**3)
#    V_exp = img3D.sum()
#    print('Vol_the=', V_the)
#    print('Vol_exp=', V_exp)
#    print('err=', V_the-V_exp)


#==============================================================================
# 
#==============================================================================

#    Imin, Imax = -1, 1
#    Rx, Ry = 11, 5
#    Rx = np.round(Rx)
#    Ry = np.round(Ry)  
#    
#     
#    rx = np.linspace(-Rx, Rx, 2*Rx + 1)   
#    ry = np.linspace(-Ry, Ry, 2*Ry + 1) 
#    [XX, YY] = np.meshgrid(rx,ry)
#    rho = np.sqrt(XX**2 + YY**2)
#    phi = np.arctan2(XX, YY)
##    rho = np.round(rho)   
#    y = 1.0/np.sqrt((np.cos(phi)**2/Rx**2) + (np.sin(phi)**2/Ry**2) )
#    
#    
##    r_half = r1
#    rho[rho<=Rx] = Imax
#    rho[rho>Rx] = Imin 
#    
#    
#    plt.imshow(y, cm.Greys_r, interpolation='nearest')
#    plt.show()
#    plt.imshow(rho, cm.Greys_r, interpolation='nearest')
#    print rho[5,:]
#    print y[5,:]

#==============================================================================
#   Circle
#==============================================================================

#    n = 2*15+1
#    r = 15
#    Imin, Imax = -1, 1
#    a, b = n//2, n//2    
#    
#    y,x = np.ogrid[-a:n-a, -b:n-b]
#    mask = x*x + y*y <= np.round(r*r)
#    
#    array = Imin*np.ones((n, n))
#    array[mask] = Imax
#
#    plt.imshow(array, cm.Greys_r, interpolation='nearest')
#    plt.show()

#==============================================================================
# 
#==============================================================================
#    n = 2*15+1
#    r = 15
#    Imin, Imax = -1, 1
#    a, b = n//2, n//2    
#    
#    y,x = np.ogrid[-a:n-a, -b:n-b]
#    rho = np.sqrt(x**2 + y**2)
#    phi = np.arctan2(x, y)    
#    mask = rho <= np.round(r)
#    
#    array = Imin*np.ones((n, n))
#    array[mask] = Imax
#
#    plt.imshow(array, cm.Greys_r, interpolation='nearest')
#    plt.show()
#==============================================================================
# Ellipse
#==============================================================================
#    n = 2*15 + 1
#    Rx, Ry = 15, 3
#    Imin, Imax = -1, 1
#    a, b = n//2, n//2    
#    
#    y, x = np.ogrid[-a:n-a, -b:n-b]
#    mask = (x*x/float(Rx**2) + y*y/float(Ry**2)) <= 1.0
#    
#    array = Imin*np.ones((n, n))
#    array[mask] = Imax
#
#    plt.imshow(array, cm.Greys_r, interpolation='nearest')
#    plt.show()





#==============================================================================
#   Sine2d
#==============================================================================
#    Sine2D = get_Sine2D(A = 1.0, OffSet=0.0, T=5,  N=50)
#
#    fig, axs = plt.subplots(1,1)
#    ax = axs
#    ax.imshow(Sine2D,  cm.Greys_r, interpolation='nearest') 
#    plt.show()

#==============================================================================
#   Sine3D
#==============================================================================
#    Sine3D = get_Sine3D(A = 1.0, OffSet=0.0, T=5,  N=50)
#
#    img3D = Sine3D
#    ny, nx, nz = img3D.shape 
#    ny2, nx2, nz2 = ny//2, nx//2, nz//2
#    x = np.linspace(-nx2, +nx2, nx)
#    y = np.linspace(-ny2, +ny2, ny)
#    z = np.linspace(-nz2, +nz2, ny)
#    
#   
#    fig, axs = plt.subplots(1,1)
#    ax = axs
#    ax.imshow(img3D[:,:,nz2],  cm.Greys_r, interpolation='nearest') 
#    plt.show()
    
#==============================================================================
#     
#==============================================================================
#    fig = plt.figure()
#    ax = fig.add_subplot('111', projection='3d')
##    ax.voxels(img3D)
#    alpha = 0.5
#    colors = [1, 1, 1, alpha]
#    ax.voxels(img3D, facecolors=colors)

#==============================================================================
# 
#==============================================================================   
#    fig, axs = plt.subplots(2,2)
#    
#    ax = axs[0,0]
#    ax.imshow(img,  cm.Greys_r, interpolation='nearest') 
#    ax.hlines(y=ny2, xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='b')
#    ax.vlines(x=nx2, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r')
#    
#
#    #Plot across middle Y
#    ax = axs[0,1]
#    ax.plot(y, img[:, nx2], color='r')
#    ax.hlines(y=0, xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='k')
#    ax.vlines(x=0, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#            
#    #Plot across middle X
#    ax = axs[1,0]
#    ax.plot(x, img[ny2, :],color='b') 
#    ax.hlines(y=0, xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='k')
#    ax.vlines(x=0, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#     
#    #Plot across middle X
#    ax = axs[1,1]
#    ax.plot(x, img) 
#    ax.hlines(y=0, xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='k')
#    ax.vlines(x=0, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='k')
#     
#    
#==============================================================================
# 
#==============================================================================
#   def get_Sphere(r1, r2, Imin=-1.0, Imax=+1.0):  
#    r1 = get_Odd(r1)
#    r2 = get_Odd(r2)
#    
#    r_half = r2 
#    r = np.linspace(-r_half, r_half, r2)
#    
#    [XX, YY, ZZ] = np.meshgrid(r, r, r)
#    rho = np.sqrt(XX**2 + YY**2 + ZZ**2)
#    rho = np.round(rho)
#    
#    r_half = r1 
#    rho[rho<r_half] = Imax
#    rho[rho>=r_half] = Imin
#    
#    return rho         
        
        
        
        
        
        