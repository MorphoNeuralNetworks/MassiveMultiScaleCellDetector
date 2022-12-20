# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:15:33 2020

@author: pc
"""

import numpy as np


from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


import sys
sys.path.append('../')
from ImageProcessing.PatchExtractor import get_ImagePatch



#==============================================================================
# 
#==============================================================================

def plot_DetectedCells(imgIn, df_Cells):
    
    #Initialization:
    #Get the number of cells detected  
    nc = df_Cells.shape[0]

#    n = int(np.ceil(np.sqrt(nc)))

    #Initialization: Set the Dimensions of the Figure as a function of the number of spatial scales
    ny, nx = nc, 3
    m = 0.75
    fig, axs = plt.subplots(ny,nx)
    graphSize = [6.0, 4.0]
    graphSize = [4.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
    fig.set_size_inches(graphSize)

    x, y, z, s = df_Cells['X'].values, df_Cells['Y'].values, df_Cells['Z'].values, df_Cells['S'].values
    k = 3
    interMethod = 'nearest'
    interMethod = 'bicubic'
#    interMethod = 'lanczos'
    for i in range(0, nc):
        
        xx, yy, zz, ss = int(x[i]), int(y[i]), int(z[i]), int(s[i])        
        dx, dy, dz = k*ss, k*ss, k*ss
        imgPatch = get_ImagePatch(imgIn, xx, yy, zz, dx, dy, dz)
        ny, nx, nz = imgPatch.shape
        
#        print('')
#        print('Center Intensity')
#        print(imgPatch[ny//2, nx//2, nz//2])
        
        #Lateral
        if nc>1:
            ax = axs[i,0]
        else:
            ax = axs[0]
        plotTitle = r'$R_{yz}=$' +  '{:0.1f}'.format(ss)
        ax.set_title(plotTitle)
        img = imgPatch[:, nx//2,:]
        img = np.max(imgPatch,axis=1)
#        img = np.mean(imgPatch,axis=1)
        ax.imshow(img, cm.Greys_r, interpolation=interMethod, origin='lower')
        C = plt.Circle((nx//2, nz//2), ss, color='b', fill=False)
        ax.add_artist(C)     
        
        #Front
        if nc>1:
            ax = axs[i,1]
        else:
            ax = axs[1] 
        plotTitle = r'$R_{xy}=$' +  '{:0.1f}'.format(ss)
        ax.set_title(plotTitle)
        img = imgPatch[:, :,nz//2]
        img = np.mean(imgPatch,axis=2)
#        img = np.mean(imgPatch,axis=2)
        ax.imshow(img, cm.Greys_r, interpolation=interMethod, origin='lower')
        C = plt.Circle((nx//2, ny//2), ss, color='b', fill=False)
        ax.add_artist(C)
        
        #Bottom
        if nc>1:
            ax = axs[i,2]
        else:
            ax = axs[2] 
        plotTitle = r'$R_{xz}=$' +  '{:0.1f}'.format(ss)
        ax.set_title(plotTitle)     
        img = imgPatch[ny//2, :, :]
        img = np.mean(imgPatch,axis=0)
#        img = np.mean(imgPatch,axis=0)
        ax.imshow(img, cm.Greys_r, interpolation=interMethod, origin='lower')
        C = plt.Circle((nz//2, ny//2), ss, color='b', fill=False)
        ax.add_artist(C)        
        

    
    fig.tight_layout(h_pad=1.0) 
    plt.show() 
#==============================================================================
# 
#==============================================================================
def plot_3DResults(imgIn, df_Cells):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #Plot: MIPs
    plot3D_3DMIP(ax, imgIn) 
    
    #Plot Cells
    xyzs = df_Cells['X'].values, df_Cells['Y'].values, df_Cells['Z'].values, df_Cells['S'].values
    plot3D_Scatter(ax, xyzs)
    
#    plt.show()
    return fig, ax
    

def plot3D_3DMIP(ax, img3D):
    ny, nx, nz = img3D.shape     
    n = np.max([nx,ny,nz])
    
    k = 2.0
    
    MIPy = np.max(img3D,axis=0)
    MIPx = np.max(img3D,axis=1)
    MIPz = np.max(img3D,axis=2)
    

    x = np.arange(0, nx, 1)
    z = np.arange(0, ny, 1)
    y = np.arange(0, nz, 1)
    
    ny, nx, nz = n, n, n
    
    #XY Plane: Front 
    X, Z = np.meshgrid(x, z)
#    ax.contourf(X, MIPz, Z, offset=nz, zdir='y', cmap=cm.jet) 
    ax.contourf(X, MIPz, Z, offset=k*nz, zdir='y', cmap=cm.jet) 
    
    #YZ Plane: Lateral
    Y, Z = np.meshgrid(y, z)
    ax.contourf(MIPx, Y, Z, offset=-nx, zdir='x', cmap=cm.jet)
    
    #XZ Plane: Bottom
    X, Y = np.meshgrid(x, y) 
    ax.contourf(X, Y, MIPy.T, offset=-ny, zdir='z', cmap=cm.jet)  
    
    #Set Axis Limits
    
    ax.set_xlim(-nx, k*nx)
    ax.set_ylim(-nz, k*nz)
    ax.set_zlim(-ny, k*ny)

    
    #Set Axis Label
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')


def plot3D_Scatter(ax, xyzs):
    x, y, z, s = xyzs
    dpi = 100.0
    pts = 72./dpi
    
    S = (s*pts)**3
#    s = np.sqrt(np.pi/S)
#    S = np.pi*(s*pts)**2
#    S = ((4.0/np.pi)*s*pts)**2
#    S = (1./6.)*np.pi*(2*s*pts)**3
#    S = (4./3.)*np.pi*(2*s*pts)**2
#    S = (1./6.)*np.pi*(s*pts)**3
    #Plot 3D Scatter
    ax.scatter(x, z, y, c='k', s=S, depthshade=0)
    
#    y0 = ax.axes.get_ylim()[1]
#    x0 = ax.axes.get_ylim()[0]
#    z0 = ax.axes.get_ylim()[0]
#    
##    #Front
#    ax.scatter(x, y0, y, c='g', depthshade=0)
##    #Lateral
#    ax.scatter(x0, z, y, c='g', depthshade=0)
##    #Bottom
#    ax.scatter(x, z, z0, c='g', depthshade=0) 
    



#==============================================================================
# 
#==============================================================================

def plot_2DResult(imgIn, df_Cells):
    
    #Initialization: Set the Dimensions of the Figure as a function of the number of spatial scales
    ny, nx = 2, 3
    m = 0.75
    fig, axs = plt.subplots(ny, nx)
    graphSize = [6.0, 4.0]
    graphSize = [4.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
    fig.set_size_inches(graphSize) 
    
    #Initialization
    imgDim = imgIn.shape
    
    
    #Ploting: Input MIP (clean plot)
    ax = axs[0,:]
    xyzMIP = get_xyzMIP(imgIn)
    plot_xyzMIP(ax, xyzMIP)
    
    #Ploting: Input MIP with Detected Cells
    ax = axs[1,:]
    xyzMIP = get_xyzMIP(imgIn)
    plot_xyzMIP(ax, xyzMIP)
    
    #Ploting: Local Maxima       
#    ax = axs[1,:]
#    xyz = df_Cells['Y'].values, df_Cells['X'].values, df_Cells['Z'].values
#    plot_LocalMaxPosition(ax, xyz, imgDim)
    
    #Ploting: Circles
    ax = axs[1,:]
    xyzsi = df_Cells['X'].values, df_Cells['Y'].values, df_Cells['Z'].values, df_Cells['S'].values, df_Cells.index.values
    plot_LocalMaxCircles(ax, xyzsi, imgDim)
        

    return fig, axs
    
    
#==============================================================================
# 
#==============================================================================
def plot_MultiScaleAnalysis(imgIn, scales, imgOutMS, df_MaxMS):
    
    #Initialization: #Get the number of Spatial Scales that were analized  
    ns = scales.shape[0]
    
    #Initialization: Set the Dimensions of the Figure as a function of the number of spatial scales
    ny, nx = 2*ns, 3
    m = 0.75
    fig, axs = plt.subplots(ny,nx)
    graphSize = [6.0, 4.0]
    graphSize = [4.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
    fig.set_size_inches(graphSize) 
    
    #Initizalization:
    even_index = np.arange(0, 2*ns, 2)    
    odd_index = even_index + 1  
    
    #Initilization:
    imgDim = imgIn.shape
    
    #Routine: Visualize the Results at each Spatial Scale
    for i in range(0,ns):
        #Get imgOut at each Spatial Scale
        imgOut = imgOutMS[i]
        scale = scales[i]
        df_Max = df_MaxMS.loc[(df_MaxMS['S'] == scale)] 
        
        
        #Ploting: Output MIP
        ax = axs[even_index[i],:]
        xyzMIP = get_xyzMIP(imgOut)
        plot_xyzMIP(ax, xyzMIP, scale)
        
        #Ploting: Local Maxima       
        ax = axs[even_index[i],:]
        xyz = df_Max['X'].values, df_Max['Y'].values, df_Max['Z'].values
        plot_LocalMaxPosition(ax, xyz, imgDim)
        
        #Ploting: Circles
        ax = axs[even_index[i],:]
        xyzsi = df_Max['X'].values, df_Max['Y'].values, df_Max['Z'].values, df_Max['S'].values, df_Max.index.values
        plot_LocalMaxCircles(ax, xyzsi, imgDim)


        #Ploting: MiddleSection
        ax = axs[odd_index[i], :]
        ymax = 1.1*df_MaxMS['I_DoG'].max()
        plot3D_MiddleSections(ax, imgIn, imgOut, scale, ymax)
        # ax.set_ylim([-2.5, +2.5])
        

    
     
    fig.tight_layout(h_pad=1.0)  
    plt.show()
    
#    my_path = os.path.dirname(sys.argv[0])
#    my_NewFolder = 'TempTest' 
#    path_NewFolder = os.path.join(my_path,  my_NewFolder)
#    if not os.path.exists(path_NewFolder):
#        os.makedirs(path_NewFolder)            
#    
#    #Saving the plot 
##    BasicName = 'scipy_signal_fftconvolve' 
#    BasicName = 'numpy_fft' 
#    my_str = BasicName  + '.png'
#    graph_dpi = 150
#    path_file = os.path.join(path_NewFolder, my_str)  
#    fig.savefig(path_file, dpi=graph_dpi, bbox_inches='tight')
    

def plot_xyzMIP(axs, xyzMIP, scale=''):
#    MIPx, MIPy, MIPz = xyzMIP[0], xyzMIP[1], xyzMIP[2] 
    MIPx, MIPy, MIPz = xyzMIP
    
    #Front   
    ax = axs[1] 
    if not scale:
        myTitle = ''
    else:     
        myTitle = r'$R_{xy}=$' +  '{:0.1f}'.format(scale) + ' (Front)'
    plot_MIP(ax, MIPz,  title=myTitle, xlabel='X', ylabel='Y')
    
    #Lateral
    ax = axs[0] 
    if not scale:
        myTitle = ''
    else: 
        myTitle = r'$R_{zy}=$' +  '{:0.1f}'.format(scale) + ' (Lateral)'
    plot_MIP(ax, MIPx, title=myTitle, xlabel='Z', ylabel='Y')

    #Bottom
    ax = axs[2]
    if not scale:
        myTitle = ''
    else: 
        myTitle = r'$R_{zx}=$' +  '{:0.1f}'.format(scale) + ' (Bottom)'
    plot_MIP(ax, MIPy.T, title=myTitle, xlabel='X', ylabel='Z')  


def plot_LocalMaxPosition(axs, xyz, imgDim):
    #jet, hot, seismic
    ny, nx, nz = imgDim
    x, y, z = xyz
    
    #Front
    ax = axs[1]
    p = ax.scatter(x, y, c=z, cmap='jet', vmin=0, vmax=nz, edgecolors='k', alpha=1.0) 
#    plt.colorbar(p, ax=ax)
 
    #Lateral
    ax = axs[0]
    p = ax.scatter(z, y, c=x, cmap='jet', vmin=0, vmax=nx, edgecolors='k', alpha=1.0) 
#    plt.colorbar(p, ax=ax)
    
    #Bottom
    ax = axs[2]
    p = ax.scatter(x, z, c=y, cmap='jet', vmin=0, vmax=ny, edgecolors='k', alpha=1.0) 
#    plt.colorbar(p, ax=ax)

#import matplotlib.colors as colors
#import matplotlib.cm as cm
#jet = colors.Colormap('jet')
#cNorm  = colors.Normalize(vmin=0, vmax=100)
#a = cm.ScalarMappable(norm=cNorm, cmap=jet)


def plot_LocalMaxCircles(axs, xyzsi, imgDim):
    x, y, z, s, ix = xyzsi
    ny, nx, nz = imgDim
    nxyz_max = np.max([ny,nx,nz])
    n = x.shape[0]    
    charSize = 12    
     
    dz = np.arange(0,nxyz_max)    
    myColors = get_colors(dz, plt.cm.jet)
#    myColors = 'b'
    
    for i in range(0,n):
        myText = str(ix[i])
        
        #Front        
        ax = axs[1] 
        c = myColors[int(z[i])]
        c = 'b'
        C = plt.Circle((x[i], y[i]), s[i], color=c, fill=False)
        ax.add_artist(C) 
        
#        print myColors[int(z[i])]
        
        ax.text(x[i], y[i], myText,
             fontsize=charSize,
             color='k',
#             color=myColors[int(z[i])],
             horizontalalignment='center',
             verticalalignment='center',
             )
    
        #Lateral
        ax = axs[0]
        c = myColors[int(x[i])]
        c = 'b'
        C = plt.Circle((z[i], y[i]), s[i], color= c, fill=False, alpha=0.5)
        ax.add_artist(C)
        ax.text(z[i], y[i], myText,
             fontsize=charSize,
             color='k',
#             color=myColors[int(x[i])],
             horizontalalignment='center',
             verticalalignment='center', 
             )
    
        #Bottom    
        ax = axs[2] 
        c = myColors[int(y[i])]
        c = 'b'
        C = plt.Circle((x[i], z[i]), s[i], color= c, fill=False)
        ax.add_artist(C)
        ax.text(x[i], z[i], myText,
             fontsize=charSize,
             color='k',
#             color=myColors[int(y[i])],
             horizontalalignment='center',
             verticalalignment='center', 
             )
        
   
def plot_MIP(ax, MIP, title='', xlabel='', ylabel=''):
    ax.set_title(title)
    ax.imshow(MIP, cmap=cm.Greys_r,  interpolation='nearest', origin='lower')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


#Maximum Intensity Projection across 3-Dimensions
def get_xyzMIP(img3D):
    ny, nx, nz = img3D.shape    
    MIPy = np.max(img3D,axis=0)
    MIPx = np.max(img3D,axis=1)
    MIPz = np.max(img3D,axis=2) 
    xyzMIP = [MIPx, MIPy, MIPz ]
    return  xyzMIP  

def plot3D_MiddleSections(axs, imgIn, imgOut, scale, ymax): 
#    print('--------------')    
#    print('', imgIn.shape)
#    print('', imgOut.shape)
    ny, nx, nz = imgIn.shape    
    x2_In = imgIn[ny//2, :, nz//2]
    y2_In = imgIn[:, nx//2, nz//2]
    z2_In = imgIn[ny//2, nx//2, :]
    
    x2_Out = imgOut[ny//2, :, nz//2]
    y2_Out = imgOut[:, nx//2, nz//2]
    z2_Out = imgOut[ny//2, nx//2, :]
    

    ax = axs[1]
    plotTitle = r'$R_{x}=$' +  '{:0.1f}'.format(scale)
    ax.set_title(plotTitle)
    plot_MiddleSection(ax, x2_In, x2_Out)
    ax.set_ylim([-ymax, +ymax])
  
    ax = axs[0]
    plotTitle = r'$R_{y}=$' +  '{:0.1f}'.format(scale)
    ax.set_title(plotTitle)
    plot_MiddleSection(ax, y2_In, y2_Out)
    ax.set_ylim([-ymax, +ymax])
   
    ax = axs[2]
    plotTitle = r'$R_{z}=$' +  '{:0.1f}'.format(scale)
    ax.set_title(plotTitle)
    plot_MiddleSection(ax, z2_In, z2_Out)
    ax.set_ylim([-ymax, +ymax])

       
def plot_MiddleSection(ax, xIn, xOut):
    n = xIn.shape[0]
    ax.plot(xIn,  color= 'k', label='In')
    ax.plot(xOut, color= 'g', label='Out')
    
    ax.set_xlim([0, n])
    ax.set_ylim([-2.5, +2.5])
#    ax.set_aspect('auto')
    ax.hlines(y=0,     xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='0.90')
    ax.vlines(x=n//2, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='0.90')
    ax.legend()




#==============================================================================
#   
#==============================================================================

def plot3D_2DCrossSection(img3D, axs, Imin=None, Imax=None):
    ny, nx, nz  = img3D.shape
#    print(ny,nx,nz)
    
    #Lateral
    ax = axs[0]
    img2D = img3D[:, nx//2,:]
    ax.imshow(img2D, cm.Greys_r, interpolation='nearest', origin='lower', vmin=Imin, vmax=Imax)
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    
    #Front
    ax = axs[1]
    img2D = img3D[:, :, nz//2]
    ax.imshow(img2D, cm.Greys_r, interpolation='nearest', origin='lower', vmin=Imin, vmax=Imax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    #Bottom
    ax = axs[2]
    img2D = img3D[ny//2, :,:]
    ax.imshow(img2D.T, cm.Greys_r, interpolation='nearest', origin='lower', vmin=Imin, vmax=Imax)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')    

#==============================================================================
# 
#==============================================================================
def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))




#==============================================================================
#     Saving Results
#==============================================================================


def save_Results(imgIn, df_Cells, rootPath, folderPath, fileName):
    import os
 
    
    my_path = os.path.dirname(sys.argv[0])
    my_NewFolder = rootPath 
    path_NewFolder = os.path.join(my_path,  my_NewFolder, folderPath)
    if not os.path.exists(path_NewFolder):
        os.makedirs(path_NewFolder)            
    
    #1) Ploting: 2D-MIPS
    fig, axs = plot_2DResult(imgIn, df_Cells)
    fig.tight_layout(h_pad=1.0)  
    plt.show()
    
    #Saving the plot 
    
    BasicName = 'Results'+ fileName +'_plot_2DResult' 
    my_str = BasicName  + '.png'
    graph_dpi = 150
    path_file = os.path.join(path_NewFolder, my_str)  
    fig.savefig(path_file, dpi=graph_dpi, bbox_inches='tight')
    
    #2) Ploting: 3D-MIPS
    fig, axs = plot_3DResults(imgIn, df_Cells)
    plt.show()
    
    #Saving the plot 
    BasicName = 'Results'+ fileName +'_plot_3DResults' 
    my_str = BasicName  + '.png'
    graph_dpi = 150
    path_file = os.path.join(path_NewFolder, my_str)  
    fig.savefig(path_file, dpi=graph_dpi, bbox_inches='tight')


    #Saving the Table
    BasicName = 'Table'+ fileName
    my_str = BasicName  + '.csv'
    path_file = os.path.join(path_NewFolder, my_str)  
    df_Cells['I'] = np.round(df_Cells['I'].values, 2)
    df_Cells.to_csv(path_file, sep=';', encoding='utf-8', index=True)  
    
 



#==============================================================================
#    
#==============================================================================
def plot_2DResultTensor(imgIn, df_Cells, overlap):
        
    #Initialization: Set the Dimensions of the Figure as a function of the number of spatial scales
    ny, nx = 1, 3
    m = 0.75
    fig, axs = plt.subplots(ny, nx)
    graphSize = [6.0, 4.0]
    graphSize = [4.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
    fig.set_size_inches(graphSize) 
    
    #Initialization
    imgDim = imgIn.shape
    
    #Ploting: Input MIP with Detected Cells
    ax = axs
    xyzMIP = get_xyzMIP(imgIn)
    plot_xyzMIP(axs, xyzMIP)
    
    #Ploting: Circles
    ax = axs
    xyzsi = df_Cells['X'].values, df_Cells['Y'].values, df_Cells['Z'].values, df_Cells['S'].values, df_Cells.index.values
    plot_LocalMaxCircles(ax, xyzsi, imgDim)
      
    #Ploting: Orientation
    ori = df_Cells['Vx'].values, df_Cells['Vy'].values, df_Cells['Vz'].values
    plot_LocalOrientations(ax, xyzsi, ori)
    
    #Ploting: Overlap    
    print()
    print( 'Ploting: Overlap')
    print(overlap)
    print(imgDim)
    plot_Overlap(axs, overlap, imgDim)
    
    return fig, axs


def plot_Overlap(axs, overlap, imgDim):
    ny, nx, nz = imgDim
    kx, ky, kz = overlap
  

    ax = axs[0]
    ax.hlines(y=kz,     xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='r', alpha=0.5)
    ax.hlines(y=nz-kz,  xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='r', alpha=0.5)
    ax.vlines(x=ky,     ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r', alpha=0.5)
    ax.vlines(x=ny-ky,  ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r', alpha=0.5)

    ax = axs[1]
    ax.hlines(y=kx,     xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='r', alpha=0.5)
    ax.hlines(y=nx-kx,  xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='r', alpha=0.5)
    ax.vlines(x=ky,     ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r', alpha=0.5)
    ax.vlines(x=ny-ky,  ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r', alpha=0.5)

    ax = axs[2]
    ax.hlines(y=kx,     xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='r', alpha=0.5)
    ax.hlines(y=nx-kx,  xmin=ax.axes.get_xlim()[0], xmax=ax.axes.get_xlim()[1], linestyle='--', color='r', alpha=0.5)
    ax.vlines(x=kz,     ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r', alpha=0.5)
    ax.vlines(x=nz-kz,  ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--', color='r', alpha=0.5)

        
    
    
def plot_LocalOrientations(axs, xyzsi, ori):
    
    x, y, z, s, ix = xyzsi
    vx, vy, vz = ori
    n = x.shape[0]
    
    for i in range(0,n):     
        x0, y0, z0 = x[i], y[i], z[i]
        vx0, vy0, vz0 = s[i]*vx[i], s[i]*vy[i], s[i]*vz[i]   

        ax = axs[0]
        ax.quiver(z0, y0, vz0, vy0,  units = 'xy', scale = 1, color='r')
        
        ax = axs[1]
        ax.quiver(x0, y0, vx0, vy0,  units = 'xy', scale = 1, color='r')
        
        ax = axs[2]
        ax.quiver(x0, z0, vx0, vz0,  units = 'xy', scale = 1, color='r')

#        ax = axs[0]
#        ax.quiver(z0, y0, v[2], v[1],  units = 'xy', scale = 2, color='r')
#        
#        ax = axs[1]
#        ax.quiver(x0, y0, v[0], v[1],  units = 'xy', scale = 2, color='r')
#        
#        ax = axs[2]
#        ax.quiver(x0, z0, v[0], v[2],  units = 'xy', scale = 2, color='r')
    


#==============================================================================
#   Results: Visualization Intermediate Results
#==============================================================================
def plot_InterMediateResult(imgIn, df_All, df_Cells0, df_Cells1, df_Cells2, df_Cells3, overlap):

        n_decimals = 2
        print('')
        print('-------------------------------------------------------')
        print('-----------0) MultiScale Detection Algorithm------------')
        print('-------------------------------------------------------') 
        df_Cells = df_All
        fig, axs = plot_2DResult(imgIn, df_Cells)
        fig.tight_layout(h_pad=1.0)  
        plt.show() 
        print('')
        print('N_cells', df_Cells.shape[0])
        print('Table:')
        print(np.round(df_Cells, n_decimals))
        
        print('')
        print('-------------------------------------------------------')
        print('-----------0) After Cell Detection Algorithm-----------')
        print('-------------------------------------------------------') 
        df_Cells = df_Cells0
        fig, axs = plot_2DResult(imgIn, df_Cells)
        fig.tight_layout(h_pad=1.0)  
        plt.show() 
        print('')
        print('N_cells', df_Cells.shape[0])
        print('Table:')
        print(np.round(df_Cells, n_decimals))
    
        # print('')
        # print('-------------------------------------------------------')
        # print('----------1) After Intensity Threholding---------------')
        # print('-------------------------------------------------------')  
        # df_Cells = df_Cells1
        # fig, axs = plot_2DResult(imgIn, df_Cells)
        # fig.tight_layout(h_pad=1.0)  
        # plt.show() 
        # print('')
        # print('N_cells', df_Cells.shape[0])
        # print('Table:')
        # print(np.round(df_Cells, n_decimals))

        # print('')
        # print('-------------------------------------------------------')
        # print('----------2) After Scale Thresholding---------------')
        # print('-------------------------------------------------------')  
        # df_Cells = df_Cells2
        # fig, axs = plot_2DResult(imgIn, df_Cells)
        # fig.tight_layout(h_pad=1.0)  
        # plt.show() 
        # print('')
        # print('N_cells', df_Cells.shape[0])
        # print('Table:')
        # print(np.round(df_Cells, n_decimals))
        
        # print('')
        # print('-------------------------------------------------------')
        # print('----------3) After Tensor Algorithm--------------------')
        # print('-------------------------------------------------------') 
        # df_Cells = df_Cells3
        # fig, axs = plot_2DResultTensor(imgIn, df_Cells, overlap)
        # fig.tight_layout(h_pad=1.0)  
        # plt.show() 
        # print('')
        # print('N_cells', df_Cells.shape[0])
        # print('Table:')
        # print(np.round(df_Cells, n_decimals))

if __name__== '__main__': 
      
    pass

    dz = np.arange(0,10)    
    c1 = get_colors(dz, plt.cm.jet)  

    dz = np.arange(0,100)    
    c2 = get_colors(dz, plt.cm.jet)      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    