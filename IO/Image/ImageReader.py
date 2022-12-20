# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:40:12 2020

@author: pc
"""
#Maths Operation
import numpy as np

#Path Handeling
from glob import glob
from pathlib import Path, PurePosixPath

#Image Library
# import cv2
import tifffile


#Ploting Library
from matplotlib import pyplot as plt
import matplotlib.cm as cm

#operative Systems Libraries       
import os
import sys
import time

import pandas as pd


import itertools
import threading


from IO.Files.FileManager import createFolder
from IO.Files.FileWriter import save_Figure
#==============================================================================
# Load Packager that are above this folder
#==============================================================================
# current_path = Path(os.path.dirname(sys.argv[0]))
# sys.path.append((str(current_path.parent.parent))) 
 
from ParallelComputing.WorkManager import multithreading, plot_ComputingPerformance


#==============================================================================
#     
#==============================================================================
def plot_ComputingPerformance2(start, stop, title_label=None):  
    
    #Figure: Settings
    ny, nx = 1, 1
    m = 1.0    
    fig, ax = plt.subplots(ny,nx)
    graphSize = [7.0, 4.0]
    graphSize = m*nx*graphSize[0], m*ny*graphSize[1] 
    inches_per_second = 60.84
    inches_per_task = 0.30
    graphSize = np.max(stop)*inches_per_second, stop.shape[0]*inches_per_task    
    fig.set_size_inches(graphSize)    
    

    #Ploting Bars
    plot_TaskRunTimes(ax, start=start, stop=stop, bar_color='b', label=None)   

    #Settings: 
    n_Tasks = start.shape[0]   
    ax.set_yticks(np.arange(0, n_Tasks + 1)) 
    ax.set_ylim(-0.5, n_Tasks + 1.5)
    ax.set_xlim(0, ax.axes.get_xlim()[1])
              
    ax.legend(loc='upper center', bbox_to_anchor=(0., +0.95, 1., 0.),
              borderaxespad=0, ncol=2, mode=None) #mode='expand'
    
    return fig, ax


def plot_TaskRunTimes(ax, start, stop, bar_color, label=None): 
    #Plot the Time that takes each Task      
    widths   = stop - start
    xcenters = start + widths/2.0  
    ycenters = np.arange(1, len(start) + 1)
    ax.barh(ycenters, widths, left=start, color=bar_color, align='center', alpha=0.5, label=label)
    for xcenter, ycenter, width in zip(xcenters, ycenters, widths):
        ax.text(xcenter, ycenter, '{:0.2f}'.format(width), ha='center', va='center', color='k')
    
    #Plot the Time that takes all Tasks      
    width = stop[-1]-start[0] 
    xcenter = start[0] + width/2.0
    ycenter = 0
    left = start[0]
    ax.barh(ycenter, width, left=left, color='0.5', align='center', alpha=0.5)
    ax.text(xcenter, ycenter, '{:0.2f}'.format(width), ha='center', va='center', color='k')
        
    return stop[-1]-start[0]   
 
#==============================================================================
# 
#==============================================================================

#Observation
#1) Only supported for 3D Image Sequence store in the same RootFolder
#2) Only supported for image with '.tif'

def read_Image(imgPath, nThreads=1):
    imgDimXYZ, bitDepth, fileExtension, memSize = get_ImageInfo(imgPath)
    coordinates = imgDimXYZ//2
    dissectionSize = imgDimXYZ + 1
    imgPatch, coordinates, dissectionSize, bitDepth, start, stop = read_ImagePatch(imgPath, coordinates, dissectionSize, nThreads=1)
    return imgPatch


def get_ImageInfo(imgPath):
    print()
    print('Start: get_ImageInfo()')
    
    #Path Information
    path = Path(imgPath)
    file_name = None
    fileExtension = PurePosixPath(path).suffix 
    # print()
    # print('imgPath:', imgPath)
    # print('path:', path)
    # print('fileExtension:', fileExtension)
    if fileExtension=='':
        rootPath = str(path)
        imgPaths =  list(path.glob('**/*.tif*'))
        # img = tifffile.imread(str(imgPaths[0]))
        # print()
        # print('rootPath:', rootPath)
        # print('imgPaths:', imgPaths)
        if imgPaths:
            img = tifffile.memmap(str(imgPaths[0])) #requires uncompressed tiff
            # img = tifffile.imread(str(imgPaths[0]))   #pip install imagecodecs
        else:
            print()
            print('Warning: Folder is empty (.tif files not found) ')
    else:
        rootPath = str(path.parent)
        file_name = path.name
        imgPaths =  str(path)
        # img = tifffile.imread(imgPaths) 
        img = tifffile.memmap(imgPaths)    
     
    # print(path) 
    # print(rootPath) 
    # print(file_name)  
    
    #Determine the Bit Depth
    dataType = img.dtype.type
    bitDepth = 0
    if dataType==np.uint8:
        bitDepth = 8
    elif dataType==np.uint16:
        bitDepth = 16
    else:
        print('BitDepth not Found')
        print('dataType:', dataType)
        
    #Read Sequence or Stack  
    imgDimXYZ = None
    img_nDim = len(img.shape)  
    if img_nDim==2:
        nz = len(imgPaths)
        ny, nx = img.shape
        imgDimXYZ = np.array([nx, ny, nz])
    elif img_nDim==3:
        nz, nx, ny = img.shape
        imgDimXYZ = np.array([nx, ny, nz])
    else:
        print('Unknow image Dimensions')
    
    #Get Image Format from the file name
    fileExtension = PurePosixPath(imgPaths[0]).suffix    
    
    #Image Memory Size in GigaBytes
    memSize = (bitDepth/8)*(np.prod(imgDimXYZ.astype(float))/10**9) # #????? Warning: data type 

    print()
    print('Stop: get_ImageInfo()')
    
    return imgDimXYZ, bitDepth, fileExtension, memSize

def read_ImagePatch(imgPath, coordinates, dissectionSize, nThreads=1, showPlot=False, t0=0, verbose=False):
    start = time.time() - t0
    
    if verbose==True:
        print()
        print('Start: read_ImagePatch()')
        print('coordinates \n', coordinates)
        print('dissectionSize \n', dissectionSize)
    
    #Casting    
    coordinates = coordinates.astype(int)
    dissectionSize = dissectionSize.astype(int)

    #Path Information
    path = Path(imgPath)
    file_name = None
    fileExtension = PurePosixPath(path).suffix    
    if fileExtension=='':
        print()
        print('-----------------------------')
        print('fileExtension', fileExtension)
        print('-----------------------------')
        print()
        rootPath = str(path)
        imgPaths =  list(path.glob('**/*.tif*'))
        # img = tifffile.imread(str(imgPaths[0]))
        # img = tifffile.memmap(str(imgPaths[0]))
        with tifffile.TiffFile(str(imgPaths[0])) as tif:
            img = tif.asarray()
    else:
        print()
        print('-----------------------------')
        print('fileExtension', fileExtension)
        print('-----------------------------')
        print()
        rootPath = str(path.parent)
        file_name = path.name
        imgPaths =  str(path)
        # img = tifffile.imread(imgPaths) 
        # img = tifffile.memmap(imgPaths) 
        with tifffile.TiffFile(imgPaths) as tif:
            img = tif.asarray()        
     

    print(path) 
    print(rootPath) 
    print(file_name)  
    print(imgPaths)      
     
    
    #Determine the Bit Depth
    dataType = img.dtype.type
    bitDepth = 0
    if dataType==np.uint8:
        bitDepth = 8
    elif dataType==np.uint16:
        bitDepth = 16
    else:
        print('BitDepth not Found')
        
    #Read Sequence or Stack     
    img_nDim = len(img.shape)  
    print(img_nDim)
    if img_nDim==2:
        imgPatch, coordinates, dissectionSize = read_ImagePatchSeries(imgPaths, coordinates, dissectionSize, nThreads, showPlot)
    elif img_nDim==3:
        imgPatch, coordinates, dissectionSize = read_ImagePatchStack(img, coordinates, dissectionSize)
    else:
        print('Unknow image Dimensions')
    
    if verbose==True:        
        print('Stop: read_ImagePatch()')
        print()
        
    stop = time.time() - t0 
    return imgPatch, coordinates, dissectionSize, bitDepth, start, stop



def read_ImagePatchSeries(imgPaths, coordinates, dissectionSize, nThreads=1, showPlot=False):
    #Unpaking    
    x,   y,  z = coordinates
    dx, dy, dz = dissectionSize 
      
    #Get 3D Image Dimensions
    [Ny, Nx, Nz] = get_DimensionsFrom3DImageSequence(imgPaths)

    #Get Extreme Index to extract the Image Patch from the Whole Image
    # x0, x1 = get_CenteredExtremes(x, dx, nx)
    # y0, y1 = get_CenteredExtremes(y, dy, ny)
    # z0, z1 = get_CenteredExtremes(z, dz, nz)
    
    #Get Extreme Index to extract the Image Patch from the Whole Image
    x0, xc, x1, nx = get_CenteredExtremes(x, dx, Nx)
    y0, yc, y1, ny = get_CenteredExtremes(y, dy, Ny)
    z0, zc, z1, nz = get_CenteredExtremes(z, dz, Nz)
    
    # print()
    # print('Extremes')    
    # print(x0, xc, x1, nx, Nx)
    # print(y0, yc, y1, ny, Ny)
    # print(z0, zc, z1, nz, Nz)

    print()
    print('coordinates', coordinates)
    print('dissectionSize', dissectionSize)
    
    # Recompute Center Coordinates and Dissection Size
    coordinates = xc, yc, zc 
    dissectionSize = nx, ny, nz
    print()
    print('coordinates', coordinates)
    print('dissectionSize', dissectionSize)
    
    #Get the paths of the zSlices
    imgPaths = imgPaths[z0:z1+1]   
    
    #Default: same number of threads as zSlides
    if nThreads=='Max':
        nThreads = len(imgPaths)
    
    #Get the paths of the slices 
    nSlices = len(imgPaths)
    taskId = np.arange(1, nSlices+1)
    myArgs = [taskId, imgPaths, itertools.repeat([x0, x1, y0, y1], len(imgPaths))]
    res = multithreading(func=read_zSlides, args=myArgs, workers=nThreads)

    
#    zSlides, start_zSlides, stop_zSlides = np.array(res).T 
    zSlides, M_Times = np.array(list(res), dtype=object).T 
    M_Times = np.concatenate(M_Times, axis=0)
    M_Times = M_Times.reshape(len(res), 6)
          
    imgPatch = np.stack(zSlides, axis=0)
    imgPatch = np.transpose(imgPatch,(2,1,0)) 
    
#    imgPatch = np.flip(imgPatch, axis=0)
#    imgPatch = np.flip(imgPatch, axis=1)
    

#    showPlot=False
    if showPlot==True: 
        #1) Unpacking: Computing Performance Data
        df_Times = pd.DataFrame(M_Times, columns=['taskID', 'compID', 'processID', 'threadID', 'start', 'stop'])
        df_Times['width'] = df_Times['stop'] - df_Times['start']
        
        fig, ax = plot_ComputingPerformance2(df_Times['start'].values, df_Times['stop'].values) 
#        fig, ax = plot_ComputingPerformance(df_Times, computation_labels=['IO_Read'])
        
        #Figure: Labels
        title_label = ('IO Performance: Reading 2D Stacks')
        ax.set_title(title_label)
        ax.set_xlabel("Time [Seconds]")
        ax.set_ylabel("Tasks [Read 2D Stacks]")    
        
        myText =    ('ParentDim='   + str(nx) + 'x' + str(ny) + 'x'  + str(nz) + '\n' +
                    'ChildDim='   + str(dx) + 'x' + str(dy) + 'x'  + str(dz) + '\n' +
                    'ChildCoord=' + str(x) + ', ' + str(y) + ', '  + str(z)  +  '\n' +
                    'nThreads=' + str(nThreads)
                    )                    
        ax.text(0.05, 0.95, myText,
             color='k',
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes)
        plt.show()
        
        #Save Figure 
        localPath = Path(os.path.dirname(sys.argv[0]))
        rootPath = localPath.parent.parent
        resultsFolder = 'Results'
        pathFolder = Path.joinpath(rootPath, resultsFolder, 'Performance_IO')
        createFolder(str(pathFolder), remove=False)
        
        
        fileName = ('ParentDim_'   + str(nx) + '_' + str(ny) + '_'  + str(nz) + 
                    '_ChildDim_'   + str(dx) + '_' + str(dy) + '_'  + str(dz) + 
                    '_ChildCoord_' + str(x) + '_' + str(y) + '_'  + str(z) +  
                    '_nT_' + str(nThreads) )
        save_Figure(fig, str(pathFolder), fileName)

    return imgPatch, coordinates, dissectionSize
#    return imgPatch, res

def read_ImagePatchStack(img3D, coordinates, dissectionSize): 
   
    #Unpaking    
    x,   y,  z = coordinates
    dx, dy, dz = dissectionSize     
     
    # img = img3D.transpose(1,2,0) 
    # [ny, nx, nz] = img3D.shape
    [Nz, Nx, Ny] = img3D.shape # ??? (read (3d.tiff) leads to nz, nx, ny
    # [ny, nx, nz] = img.shape ???

    #Get Extreme Index to extract the Image Patch from the Whole Image
    x0, xc, x1, nx = get_CenteredExtremes(x, dx, Nx)
    y0, yc, y1, ny = get_CenteredExtremes(y, dy, Ny)
    z0, zc, z1, nz = get_CenteredExtremes(z, dz, Nz)
    
    # print()
    # print('Extremes')    
    # print(x0, xc, x1, nx, Nx)
    # print(y0, yc, y1, ny, Ny)
    # print(z0, zc, z1, nz, Nz)
    
    print()
    print('coordinates', coordinates)
    print('dissectionSize', dissectionSize)
    
    # Recompute Center Coordinates and Dissection Size
    coordinates = xc, yc, zc 
    dissectionSize = nx, ny, nz
    print()
    print('coordinates', coordinates)
    print('dissectionSize', dissectionSize)  


    # imgPatch = img3D[y0:y1+1,x0:x1+1,z0:z1+1] 
    imgPatch = img3D[z0:z1+1, x0:x1+1, y0:y1+1] 
    imgPatch = np.transpose(imgPatch,(2,1,0)) 
    
#    imgPatch = np.flip(imgPatch, axis=1)

    # print('imgPatch', imgPatch.shape)
    
    return imgPatch, coordinates, dissectionSize



#==============================================================================
# 
#==============================================================================
      
def read_zSlides(taskId, imgPath, args, t0=0):   
    # print('')
    # print('read_zSlides')
    
    start = time.time() - t0
    x0, x1, y0, y1 = args
    
    #Adittional Parameters
    compId = 0
    process_ID = os.getpid()
    thread_ID = threading.current_thread().ident
    
    #Open a small Portion of each the Image
    #Op1: Using openCV (slow reading)
#    imgSlice = cv2.imread(str(imgPath), -1)

    #Op2a: Using tifffile (faster reading) two steps
    # imgSlice = tifffile.imread(str(imgPath)) #ifffile is faster than cv2
    # imgSlice = imgSlice[y0:y1+1, x0:x1+1].T 
    
    #Op2b: Using tifffile (faster reading) single step
    # imgSlice = tifffile.imread(str(imgPath))[y0:y1+1, x0:x1+1].T 
    
    #Op3: Using tifffile (faster reading) single step (saving RAM memory)
    imgSlice = tifffile.memmap(str(imgPath))[y0:y1+1, x0:x1+1].T 

    #Changing the Image Reference System
#    imgSlice = imgSlice[x0:x1+1, y0:y1+1]
#    imgSlice = np.flip(imgSlice, axis=1)
    
    stop = time.time() - t0
    op1 = [taskId, compId, process_ID, thread_ID, start, stop] 
    return imgSlice, op1

#    process_ID = os.getpid()
#    thread_ID = threading.current_thread().ident
#    print('process_ID', process_ID)
#    print('thread_ID', thread_ID)    

    
#==============================================================================
#     
#==============================================================================
    
def get_ImagePaths(rootPath, imgFormat='tif*'):
    nameFilter = '\\*.' + imgFormat
    imgPaths = (glob(rootPath + nameFilter))
    return imgPaths

def get_DimensionsFrom3DImageSequence(imgPaths):
    # print()
    # print('get_DimensionsFrom3DImageSequence')
    # print(imgPaths)
    
    # Op1: Open with OpenCV
    # img = cv2.imread(str(imgPaths[0]), -1)
    
    # Op2a: Open with tifffile
    # img = tifffile.imread(str(imgPaths[0]))
    
    # Op2b: Open with tifffile
    img = tifffile.memmap(str(imgPaths[0]))
    
    ny, nx = img.shape
    nz = len(imgPaths)
    return ny, nx, nz

def get_zerosOddMatrix(ny, nx, nz, dataType):
    nx = get_Odd(nx)
    ny = get_Odd(ny)
    nz = get_Odd(nz)
    
    M = np.zeros((ny,nx,nz), dtype=dataType)
    return M

def get_CenteredExtremes(r, dr, rmax):
    dr_half = dr//2  
    [r0, r1 ] = [r - dr_half, r + dr_half]
    
    # print('r0, r1', r0, r1)
    
    # Manage Boundaries
    if (r0<=0)&(r1>=rmax):
        r0 = 0
        # r1 = rmax - 1
        r1 = rmax
    elif (r0<=0):
        r0 = 0
        # r1 = r1 - 1
    elif (r1>=rmax):
        r0 = r0 + 1
        # r1 = rmax -1
        r1 = rmax
    
    # #???? Get Odd Extreme
    # if dr%2==0:
    #     r1 = r1-1
    
    # Get the Size
    n = (r1 - r0) + 1
    
    # Get the Center
    rc = np.round(r0 + 0.5*(n)).astype(int) 
    
    return (r0, rc, r1, n) 
    
#def get_CenteredExtremes(r, dr, rmax):
#    dr_half = dr//2  
#    rmax = rmax - 1 
#    [r0, r1 ] = [r - dr_half, r + dr_half]
#    if r0<0:
#        r0 = 0
#        r1 = 0 + 2*dr_half
#    if r1>rmax:
#        r0 = rmax - 2*dr_half
#        r1 = rmax 
#    return r0, r1

def get_Odd(num):
    if (num % 2) == 0: 
        num = num + 1
    return num
        

#==============================================================================
# 
#==============================================================================

def read_MergedImage(rootPathImg, v_xyz_ani, dissectionSize_ani, t0=0):
    
    #Get    
    xyz_origin = v_xyz_ani[0]  -  (dissectionSize_ani - 0*1)/2 
    xyz_final  = v_xyz_ani[-1]  + (dissectionSize_ani - 0*1)/2 
    dissectionSize = xyz_final - xyz_origin
    xyz_center = xyz_origin + dissectionSize//2 + 1
    
    # print(xyz_origin)
    # print(xyz_final)
    # print(xyz_center)
    # print(dissectionSize)
    
    #Read image
    [imgPatch, dataType, start, stop] = read_ImagePatch(rootPathImg, xyz_center, dissectionSize , nThreads=2, showPlot=False, t0=t0)

    return imgPatch, dataType, start, stop


#==============================================================================
# 
#==============================================================================
#        myArgs = [imgPaths, itertools.repeat([x0, x1, y0, y1], len(imgPaths))]
#        myArgs = [imgPaths, [[x0, x1, y0, y1] for i in range(len(imgPaths))]]
#    img = cv2.imread(imgPaths[0], -1)

if __name__== '__main__':
  
  
#==============================================================================
#   #Input
#==============================================================================
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\620x905x1708_2tiff_8bit'
    x, y, z = 309, 327, 850
    
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\2482x3620x1708_2tiff_8bit' 
    BrainRegion = 'mCA1'   
    x, y, z = 1238, 1310, 850
    
    
    rootPath = r'F:\Arias\BrainScans\SPIM_MuVi\SPIM_MuVi_Scans\Ref_1800_34583_SPIM_MuVi_Line_ACF_Scan\Ref_1800_34583_Raw_UnChunked_Stitched_x1\RES(35568x32216x1951)' 
    # rootPath = r'F:\Arias\BrainScans\SPIM_MuVi\SPIM_MuVi_Scans\Ref_1800_34583_SPIM_MuVi_Line_ACF_Scan\Ref_1800_34583_Raw_UnChunked_Stitched_x1\RES(35568x32216x1951)\000000_000000_039000.tif' 
    BrainRegion = 'mCA1'   
    x, y, z = 2238, 4310, 1000
    x, y, z = 0, 0, 1000
# =============================================================================
#     
# =============================================================================
    
    
    # imgPath = r'F:\Arias\BrainScans\SPIM_MuVi\SPIM_MuVi_Scans\Ref_1800_34583_SPIM_MuVi_Line_ACF_Scan\Ref_1800_34583_Raw_UnChunked_Stitched_x1\RES(35568x32216x1951)\000000_000000_039000.tif' 
    # imgPath = str(Path(imgPath))
    # mm = tifffile.memmap(imgPath)
    # img = tifffile.imread(str(imgPath))
    
    # rootPath = str(Path(rootPath))
    # imgPath = rootPath
    # img = tifffile.imread(str(imgPath))
    #Extremes Test
#    x, y, z = 0, 0, 0
#    x, y, z = 2482, 3620, 1708
# =============================================================================
#     
# =============================================================================
    dx, dy, dz = [3, 3,  3]
    dx, dy, dz = [5, 5,  5]
    dx, dy, dz = [21, 21,  21]      #1.65sec//0.99sec ->1.7 faster with Threads
    dx, dy, dz = [21, 21,  3]      #1.65sec//0.99sec ->1.7 faster with Threads
    dx, dy, dz = [1501, 1501,  1]
    
    

    x, y, z = 35568//4, 32216//4, 1200
    dx, dy, dz = [35568//2, 32216//3,  1]
    
    x, y, z = 35568//8-650, 32216//4+5000+280, 1200
    dx, dy, dz = [35568//100, 35568/100,  1]
    
    # y, x, z = 12500, 5000, 1800
    # dx, dy, dz = [4500, 4500,  1]
    
    # y, x, z = 12490, 3360, 978
    # dx, dy, dz = [200, 200,  1]
#==============================================================================
#   Alvaro  
#==============================================================================

#    rootPath = 'D:\\MyPythonPosDoc\\Brains\\Brains_Alavaro\\myTest'
#    rootPath = 'D:\\MyPythonPosDoc\\Brains\\Brains_Alavaro\\myTest'
#    rootPath = 'D:\\MyPythonPosDoc\\Brains\\Brains_Alavaro'
#    rootPath = 'D:\\MyPythonPosDoc\\Brains\\Brains_Alavaro\\Adrian cFos Foto1 Stack.tif'
#    
#    BrainRegion = 'mCA1'   
#    x, y, z = 1391, 458, 6  
##    x, y, z  = 1391 , 458, 32 #out of boundaries
#
##    dx, dy, dz = [11, 11, 11]
#    dx, dy, dz = [51, 51, 51]
##    dx, dy, dz = [51, 21, 21]
#    

#==============================================================================
#   #Get a 3D Patch from a Big 3D Image
#   #Only supported for Image Sequence given in tif format
#==============================================================================
    coordinates = np.asarray([x,y,z])
    dissectionSize = np.asarray([dx, dy, dz])   
    
    # read_ImagePatchSeries(rootPath, coordinates, dissectionSize, nThreads=1, showPlot=False)

            
#    [img, bitDepth, start, stop] = read_ImagePatch(rootPath, coordinates, dissectionSize, nThreads=1,     showPlot=True, t0=time.time())
    
    [img, bitDepth, start, stop] = read_ImagePatch(rootPath, coordinates, dissectionSize, nThreads=5, showPlot=False, t0=time.time())
#    [img, bitDepth, start, stop, res] = read_ImagePatch(rootPath, coordinates, dissectionSize, nThreads=2, showPlot=True, t0=time.time())

#    [img, bitDepth, start, stop] = read_ImagePatch2(rootPath, coordinates, dissectionSize, nThreads=1,     showPlot=True, t0=time.time())

#    [img, bitDepth, dt] = read_ImagePatch(rootPath, coordinates, dissectionSize, nThreads=2,     showPlot=True, t0=time.time())
#    [img, bitDepth, dt] = read_ImagePatch(rootPath, coordinates, dissectionSize, nThreads='Max', showPlot=True, t0=time.time())
    
    
#==============================================================================
#     Visualizations
#==============================================================================
    #Get Image Paths from the RootFolder
    imgPaths = get_ImagePaths(rootPath)
    
    #Get 3D Image Dimensions
    imgWholeShape = get_DimensionsFrom3DImageSequence(imgPaths)  
    
    #Get Patch Dimensions
    imgPatchShape = img.shape
    
    print('WholeDimensions=', imgWholeShape)
    print('bitDepth=', bitDepth)    
    print('Reading Time IO=', stop-start)
    
    #Plotting the Image
    print('')
    print('PatchDimensions=', imgPatchShape)
    [ny, nx, nz] = imgPatchShape
    imgMiddleSlice = img[:,:,nz//2]    
    plt.imshow(imgMiddleSlice,  cm.Greys_r, interpolation='nearest') 
    plt.show()
    



    # plt.imshow(img,  cm.Greys_r, interpolation='nearest') 
    # plt.show()





#==============================================================================
# Draft
#==============================================================================

# =============================================================================
#    OPTiSPIM DataSets (Workstation: Lacie) 
# =============================================================================
#    rootPath = 'K:\AriasDB\Brains\MuViSPIM\17635\2_Stitched_Nothing_dim_ny_12_nx_7_nz_1901_Subset_y_0_11_x_0_6_z_0_1900\RES(21342x12583x1810)'
#    rootPath = 'K:\\AriasDB\\Brains\\MuViSPIM\\17635\\2_Stitched_Nothing_dim_ny_12_nx_7_nz_1901_Subset_y_0_11_x_0_6_z_0_1900\\RES(21342x12583x1810)'
##    rootPath = 'K:\\AriasDB\\Brains\\MuViSPIM\\17635\\2_Stitched_Nothing_dim_ny_2_nx_2_nz_1901_Subset_y_0_1_x_3_4_z_925_974\\RES(3802x3802x38)'
##    rootPath = r'K:\AriasDB\Brains\MuViSPIM\17635\2_Stitched_Nothing_dim_ny_2_nx_2_nz_1901_Subset_y_0_1_x_3_4_z_925_974\RES(3802x3802x38)'
#    
#    x, y, z = 1238, 1310, 20
#    x, y, z = 5500, 600, 900 #MoviSPIM Cells
#    x, y, z = 1080, 400, 20 #MoviSPIM Cells
#    x, y, z = 5052, 5075, 996

# =============================================================================
#    OPTiSPIM DataSets (Workstation: Lacie) Chung
# =============================================================================
#    rootPath = 'K:\\Hemisphere_Chung\\stitched_im_series\\RES(4469x3359x843)\\000000\\000000_086684'
#    BrainRegion = 'mCA1' 
#    x, y, z =  3359, 4469, 497     
#    x, y, z =  1169, 1432, 497 
#    dx, dy, dz = [21, 21,  21]
    

   
    
    
# =============================================================================
#     MOViSPIM DataSets (Workstation: Lacie)
# =============================================================================
#    rootPath = 'K:\\Hemisphere_Chung\\stitched_im_series\\RES(4469x3359x843)\\000000\\000000_086684'
# 
#    x, y, z = np.asarray([1139, 1459, 359])
##    from pathlib import Path
##    rootPath = Path(rootPath)
##    
##    working_directory = Path(rootPath)
##    print('%r'%str(working_directory))
#    
#    str(rootPath)
#    dx, dy, dz = [151, 151,  33] 
#    dx, dy, dz = [71, 71,  71] #21sec ->depends on Z readings
#    dx, dy, dz = [31, 31,  71] #21sec
#    dx, dy, dz = [31, 31,  31] #9sec
#    dx, dy, dz = [71, 71,  31] #9sec   






#==============================================================================
#   JuanLu
#==============================================================================
#    rootPath = 'D:\\MyPythonPosDoc\\Brains\\Juanlu_ConFocal_sp8\\' 
#    dx, dy, dz = [3, 3,  3]
#    dx, dy, dz = [5, 5,  5]
#    dx, dy, dz = [21, 21,  21]
#    dx, dy, dz = [201, 201,  201]
##    dx, dy, dz = [1001, 1001,  201]
#    x, y, z = 1238, 1310, 850
#    x, y, z = 0, 0, 0
#    x, y, z = 500, 200, 15
#    imgPaths = get_ImagePaths(rootPath)
#
#
#    img = tifffile.imread(imgPaths[0]) 
#    img = img[:,:,:,1]    
#    img = img.transpose(1,2,0)
# 
#    [ny, nx, nz] = img.shape
#
#    #Get Extreme Index to extract the Image Patch from the Whole Image
#    x0, x1 = get_CenteredExtremes(x, dx, nx)
#    y0, y1 = get_CenteredExtremes(y, dy, ny)
#    z0, z1 = get_CenteredExtremes(z, dz, nz)
#    
#    imgCrop = img[y0:y1+1,x0:x1+1,z0:z1+1]
#    
#    print(imgCrop.shape)
#    
#    plt.imshow(imgCrop[:,:,15],  cm.Greys_r, interpolation='nearest') 
#    plt.show()
#    plt.imshow(img[:,:,15],  cm.Greys_r, interpolation='nearest') 
#    plt.show()