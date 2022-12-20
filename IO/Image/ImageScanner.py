# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 14:52:00 2020

@author: pc
"""
import numpy as np


#==============================================================================
# Main Routines: Scanner
#==============================================================================

def get_scanningCoordinates(scannerStart_In_px, scannerEnd_In_px, imgDimXYZ, voxelSize_In_um, voxelSize_Out_um, cellRadiusMax_um, computingCubeSize, computingCubeOverlap):     
    #Unpack args

    # =============================================================================
    #     Computing Scanner Size (isotropy)
    # =============================================================================
    
    #Voxel Ratio: In/Out
    voxelSize_Ratio_um = voxelSize_Out_um/voxelSize_In_um
    voxelSize_Ratio_px = 1.0/voxelSize_Ratio_um
    
    # Scanner Size: Forcing isotropy ??????
    mk = computingCubeSize
    scannerSize_um = mk*3*(2*cellRadiusMax_um)
    scannerSize_In_px = (np.round(scannerSize_um/voxelSize_In_um)).astype('int')
    # scannerSize_In_px = convert_to_Odd(scannerSize_In_px)
    scannerSize_Out_px = (np.round(voxelSize_Ratio_px*scannerSize_In_px)).astype('int')
    # scannerSize_Out_px = convert_to_Odd(scannerSize_Out_px)

    # =============================================================================
    #  Computing Overlap
    # =============================================================================
    kr = computingCubeOverlap
    r_max_um = kr*(2*cellRadiusMax_um)
    r_max_px = int(np.round(r_max_um/voxelSize_In_um.min()))  
    
    rx, ry = r_max_px, r_max_px
    rz = int(np.round((voxelSize_In_um[0]/voxelSize_In_um[2])*rx))
    # if dz%2==0:
    #     rz = rz + 1        
    scannerOverlap_In_px  = np.asarray([rx, ry, rz])    
    scannerOverlap_Out_px = (np.round(scannerOverlap_In_px*voxelSize_Ratio_px)).astype(int)

    # =============================================================================
    # Compute Scanner Positions
    # =============================================================================
    #Get the scanning coordinates of the Asitropy
    scannerDistance_In = get_scanningDistance(scannerSize_In_px, scannerOverlap_In_px)
    scannerPositions_In_px = get_scanningLocations(scannerStart_In_px, scannerEnd_In_px, scannerDistance_In)    
    
    print()
    print('scannerStart_In_px', scannerStart_In_px)
    print('scannerEnd_In_px', scannerEnd_In_px)
    print('scannerDistance_In', scannerDistance_In)
     
   
    print()
    print('Before')
    print('Dim:', scannerPositions_In_px.shape)
    print(scannerPositions_In_px)
    
    #Filter: Inside the Boundary 
    
    # Op1 
    # maskX = (scannerPositions_In_px[:,0] <= imgDimXYZ[0])
    # maskY = (scannerPositions_In_px[:,1] <= imgDimXYZ[1])
    # maskZ = (scannerPositions_In_px[:,2] <= imgDimXYZ[2])
    
    # # Op2
    # if scannerSize_In_px[0] <  imgDimXYZ[0]:
    #     maskX = (scannerPositions_In_px[:,0] <= imgDimXYZ[0])
    # else:
    #     maskX = (scannerPositions_In_px[:,0] <= 2*imgDimXYZ[0])
        
    # if scannerSize_In_px[1] <  imgDimXYZ[1]:
    #     maskY = (scannerPositions_In_px[:,1] <= imgDimXYZ[1])
    # else:
    #     maskY = (scannerPositions_In_px[:,1] <= 2*imgDimXYZ[1])        

    # if scannerSize_In_px[2] <  imgDimXYZ[2]:
    #     maskZ = (scannerPositions_In_px[:,2] <= imgDimXYZ[2])
    # else:
    #     maskZ = (scannerPositions_In_px[:,2] <= 2*imgDimXYZ[2])


    # # Op3
    # if scannerSize_In_px[0] <  imgDimXYZ[0]:
    #     print()
    #     print('X smaller')
    #     print('scannerSize_In_px', scannerSize_In_px)
    #     print('scannerSize_In_px[0]', scannerSize_In_px[0])
    #     print('imgDimXYZ[0]',  imgDimXYZ[0])
    #     maskX = (scannerPositions_In_px[:,0] <= (imgDimXYZ[0] + 0.5*scannerSize_In_px[0] + 1))
    # else:
    #     print('X greater')
    #     print('scannerSize_In_px', scannerSize_In_px)
    #     print('scannerSize_In_px[0]', scannerSize_In_px[0])
    #     print('imgDimXYZ[0]',  imgDimXYZ[0])
    #     maskX = (scannerPositions_In_px[:,0] <= 1*imgDimXYZ[0])
        
    # if scannerSize_In_px[1] <  imgDimXYZ[1]:
    #     maskY = (scannerPositions_In_px[:,1] <= imgDimXYZ[1] + 0.5*scannerSize_In_px[1] + 1)
    # else:
    #     maskY = (scannerPositions_In_px[:,1] <= 1*imgDimXYZ[1])        

    # if scannerSize_In_px[2] <  imgDimXYZ[2]:
    #     maskZ = (scannerPositions_In_px[:,2] <= imgDimXYZ[2] + 0.5*scannerSize_In_px[2] + 1)
    # else:
    #     maskZ = (scannerPositions_In_px[:,2] <= 2*imgDimXYZ[2]) #??? There is a bug here
    #     #Bypassing the bug
    #     z_ratio = voxelSize_In_um[2]/voxelSize_In_um[1]
    #     imgDimZ = z_ratio*imgDimXYZ[2]
    #     maskZ = (scannerPositions_In_px[:,2] <= 2*imgDimZ) #??? There is a bug here
        
    #Op4
    if (scannerSize_In_px[0] - scannerSize_In_px[0]/2) <  imgDimXYZ[0]:
        print()
        print('X smaller')
        print('scannerSize_In_px', scannerSize_In_px)
        print('scannerSize_In_px[0]', scannerSize_In_px[0])
        print('imgDimXYZ[0]',  imgDimXYZ[0])
        maskX = (scannerPositions_In_px[:,0] <= (imgDimXYZ[0] + 0.5*scannerSize_In_px[0] + 1))
    else:
        print('X greater')
        print('scannerSize_In_px', scannerSize_In_px)
        print('scannerSize_In_px[0]', scannerSize_In_px[0])
        print('imgDimXYZ[0]',  imgDimXYZ[0])
        maskX = ((scannerPositions_In_px[:,0] - scannerSize_In_px[0]/2) <= 1*imgDimXYZ[0])
    
    if (scannerSize_In_px[1] - scannerSize_In_px[1]/2) <  imgDimXYZ[1]:
        maskY = (scannerPositions_In_px[:,1] <= imgDimXYZ[1] + 0.5*scannerSize_In_px[1] + 1)
    else:
        maskY = ((scannerPositions_In_px[:,1] - scannerSize_In_px[1]/2) <= 1*imgDimXYZ[1])       

    if (scannerSize_In_px[2] - scannerSize_In_px[2]/2) <  imgDimXYZ[2]:
        maskZ = (scannerPositions_In_px[:,2] <= imgDimXYZ[2] + 0.5*scannerSize_In_px[2] + 1)
    else:
        maskZ = (scannerPositions_In_px[:,2] <= 2*imgDimXYZ[2]) #??? There is a bug here
        #Bypassing the bug
        z_ratio = voxelSize_In_um[2]/voxelSize_In_um[1]
        imgDimZ = z_ratio*imgDimXYZ[2]
        maskZ = (scannerPositions_In_px[:,2] <= 2*imgDimZ) #??? There is a bug here
       
    mask  = (maskX & maskY & maskZ)
    scannerPositions_In_px = scannerPositions_In_px[mask]
    

    print()
    print('imgDimXYZ', imgDimXYZ)
    print('voxelSize_In_um', voxelSize_In_um)
    print('voxelSize_Out_um', voxelSize_Out_um)
   
    print()
    print('After')
    print('Dim:', scannerPositions_In_px.shape)
    print(scannerPositions_In_px)
    
    #Get the scanning coordinates of the Isotropy
    scannerPositions_Out_px = (np.round(scannerPositions_In_px*voxelSize_Ratio_px)).astype(int)

    return scannerPositions_In_px, scannerSize_In_px, scannerOverlap_In_px, scannerPositions_Out_px, scannerSize_Out_px, scannerOverlap_Out_px


#Get the xyz coordinates of the Center of the Scanning Volumes by knowing...
#   1) v0: the starting point coordingate
#   2) v1: the ending point coordingate 
#   3) d : the distance between the centers of two scanning volumes
def get_scanningLocations(v0, v1, d):
    #Op1
    #nothing
    
    #Op2
    v1 = 2*v1
    
    vx = np.arange(v0[0], v1[0] + 1, d[0])
    vy = np.arange(v0[1], v1[1] + 1, d[1]) 
    vz = np.arange(v0[2], v1[2] + 1, d[2]) 
    v_xyz = get_permutationWithRepetition([vx,vy,vz])
    return v_xyz


#Get the scanning Disntance between two Scanning Volumes by knowing....
#   1) dissectionSize: the dimensions of the scanning Volume Unit
#   2) overlap: the overlap between adjacent scanning Volumes  
def get_scanningDistance(dissectionSize, overlap, mode='pixels'):
    dissectionSize = np.asarray(dissectionSize)
    overlap = np.asarray(overlap)
    
    if mode=='pixels':
        overlap = np.asarray(overlap, dtype=np.int)
        d = dissectionSize - overlap
        
    elif mode=='percentage':        
        d = (1.0 - overlap)*dissectionSize
        d = d.astype(np.int)
        
    else: 
        print('Module: IO/Image/Scanner. Error: Unsuported mode')
        
    return d

#==============================================================================
# SubRoutines: 
#==============================================================================    
#Combinatory: Permutation with Repetition
def get_permutationWithRepetition(v):
    v_comb = np.array(np.meshgrid(*v)).T.reshape(-1, len(v)) 
    return v_comb


 
#==============================================================================
#    Testing 
#==============================================================================   
if __name__== '__main__':
    
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    
    from Reader import read_ImagePatch
    
#==============================================================================
#     
#==============================================================================
#    vx = [-1, 0, 1]
#    vy = [-1, 0, 1]
#    vz = [-1, 1]
#    a = get_permutationWithRepetition([vx,vy])
#    a = get_permutationWithRepetition([vx,vy,vz])
#    
#    jajaj
#==============================================================================
#     
#==============================================================================
    #Path of the 3D image        
    rootPath = 'D:\\MyPythonPosDoc\\Brains\\2482x3620x1708_2tiff_8bit' 
    
    #Size of the Computing Unit
    nx, ny, nz = 21, 21, 21  
    nx, ny, nz = 41, 41, 41 
#    nx, ny, nz = 41, 41, 41 
#    nx, ny, nz = 61, 61, 61 
#    nx, ny, nz = 100, 100, 100 
    dissectionSize = [nx, ny, nz]
     
    #Start Point (centered) 
    BrainRegion = 'mCA1'   
#    x0, y0, z0 = 1238, 1310, 850
#    BrainRegion = 'mSub_On' 
    x0, y0, z0 = 1127, 518, 850 
    v0 = [x0, y0, z0]
    
    #End Point (centered) 
    x1, y1, z1 = x0 + 2*nx, y0, z0
#    x1, y1, z1 = x0 + 2*nx, y0 + 2*ny, z0
    v1 = [x1, y1, z1]
    
    #Overlap (as percentage)
    mode = 'percentage'            
    overlap = [0.45, 0.45, 0.45]
    overlap = [0.25, 0.25, 0.25]
    
#    #Overlap (as pixels)
    #Note: the overalp should be the radius of the bigger neuron
    mode = 'pixels'
    overlap = [12, 12, 12]
#    overlap = [22, 22, 22]
       
#==============================================================================
# Scanning Function
#==============================================================================
     
#    #Get Image Paths from the RootFolder
#    imgPaths = get_ImagePaths(rootPath)
#    
#    #Get 3D Image Dimensions
#    [Ny, Nx, Nz] = get_DimensionsFrom3DImageSequence(imgPaths) 
        
    d = get_scanningDistance(dissectionSize, overlap, mode=mode)
    v_xyz = get_scanningLocations(v0, v1, d) 
    
    #Plotting Settings:
    nny, nnx = 1, v_xyz.shape[0]
    m = 0.75
    fig, axs = plt.subplots(nny,nnx)
    graphSize = [4.0, 4.0]
    graphSize = m*nnx*graphSize[0], m*nny*graphSize[1]    
    fig.set_size_inches(graphSize)
    vmin, vmax = None, None
#    vmin, vmax = 0, 255
    for i in range(0 , v_xyz.shape[0]):        
        x, y, z = v_xyz[i]
        ax = axs[i]
        
        img = read_ImagePatch(rootPath, v_xyz[i],dissectionSize)
        imgMiddleSlice = img[:,:,nz//2]    
        ax.imshow(imgMiddleSlice,  cm.Greys_r, interpolation='nearest', vmin=vmin, vmax=vmax) 
    plt.show()

#==============================================================================
# Draft
#==============================================================================
#    xyz = np.dstack((vx,vy))

#    [x0, y0, z0] = v0 
#    [x1, y1, z1] = v1 
#    [nx, ny, nz] = imgSize
#    [x_ov, y_ov, z_ov] = overlap
#    BrainRegion = 'boundry' 
#    x0, y0, z0 = 0, 0, 0 









