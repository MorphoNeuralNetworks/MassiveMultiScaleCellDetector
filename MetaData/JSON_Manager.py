# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:23:03 2021

@author: aarias
"""


import json
from  pathlib import Path
from IO.Files.FileManager import createFolder
import numpy as np

# =============================================================================
# Visualize
# =============================================================================
def save_jsonVisualize(*args):
    
    pathFolder_ReadImage, pathFolder_WriteResults, voxelSize_In_um, voxelSize_Out_um = args
    myDic = {
            "pathFolder_ReadImage":     str(Path(str(pathFolder_ReadImage))),
            "pathFolder_WriteResults":    str(Path(str(pathFolder_WriteResults))),
            
             "voxelSize_VisIn_um"  : list(voxelSize_In_um),
             "voxelSize_VisOut_um" : list(voxelSize_Out_um)
            }      
        
    #Set Path File to save the json
    rootPath = Path(pathFolder_WriteResults)
    folderName = "Settings"
    fileName   = "Visualize.json" 
    pathFolder = Path.joinpath(rootPath, folderName)
    createFolder(str(pathFolder), remove=False)
    pathFile   = Path.joinpath(rootPath, folderName, fileName)

    print()
    print('Saving JSON: Visualize Settings')
    print(myDic)    
    
    with open(pathFile, "w") as write_file: 
        # json.dump(myDic, write_file)
        json.dump(myDic, write_file, indent=1, separators=(", ", ": "), sort_keys=False)    
    
def read_jsonVisualize(pathFile): 

    #Load Json
    with open(pathFile, "r") as read_file: 
        data = json.load(read_file)
   
    pathFolder_ReadImage = str(Path(data["pathFolder_ReadImage"])) 
    pathFolder_WriteResults = str(Path(data["pathFolder_WriteResults"])) 
    voxelSize_VisIn_um = np.array(data["voxelSize_VisIn_um"])
    voxelSize_VisOut_um = np.array(data["voxelSize_VisOut_um"])
 
    return pathFolder_ReadImage, pathFolder_WriteResults, voxelSize_VisIn_um, voxelSize_VisOut_um

# =============================================================================
# Detect
# =============================================================================
def save_jsonDetect(*args):
    
    pathFolder_ReadImage, pathFolder_WriteResults = args[0], args[1]
    xyzStart, xyzStop = args[2], args[3]
    dimXYZ = args[4]
    voxelSize_DetIn_um, voxelSize_DetOut_um = args[5], args[6]
    cellRadiusMin_um, cellRadiusMax_um = args[7], args[8]
    nScales = args[9]
    computingCubeSize, computingCubeOverlap = args[10], args[11]
    computeTensor = args[12]
    nProcess = args[13]

    myDic = {
            "pathFolder_ReadImage"      :     str(Path(str(pathFolder_ReadImage))),
            "pathFolder_WriteResults"   :     str(Path(str(pathFolder_WriteResults))),
            
            "xyzStart"  : xyzStart.tolist(),
            "xyzStop"   : xyzStop.tolist(),
            
            "dimXYZ"   : dimXYZ.tolist(),
             
            "voxelSize_DetIn_um"   : voxelSize_DetIn_um.tolist(),
            "voxelSize_DetOut_um"  : voxelSize_DetOut_um.tolist(),
             
            "cellRadiusMin_um"     : int(cellRadiusMin_um),
            "cellRadiusMax_um"     : int(cellRadiusMax_um),
           
            "nScales"              : int(nScales),
           
            "computingCubeSize"        : int(computingCubeSize),
            "computingCubeOverlap"     : int(computingCubeOverlap),
           
            "computeTensor"     : bool(computeTensor),
            
            "nProcess"     : int(nProcess), 
             
            }      
        
    #Set Path File to save the json
    rootPath = Path(pathFolder_WriteResults)
    folderName = "Settings"
    fileName   = "Detect.json" 
    pathFolder = Path.joinpath(rootPath, folderName)
    createFolder(str(pathFolder), remove=False)
    pathFile   = Path.joinpath(rootPath, folderName, fileName)
    
    print()
    print('Saving JSON: Detect Settings')
    print(myDic)
    
    with open(pathFile, "w") as write_file: 
        # json.dump(myDic, write_file)
        json.dump(myDic, write_file, indent=1, separators=(", ", ": "), sort_keys=False)    
    
def read_jsonDetect(pathFile): 

    #Load Json
    with open(pathFile, "r") as read_file: 
        data = json.load(read_file)
   
    pathFolder_ReadImage = str(Path(data["pathFolder_ReadImage"])) 
    pathFolder_WriteResults = str(Path(data["pathFolder_WriteResults"])) 
     
    xyzStart = np.array(data["xyzStart"])
    xyzStop = np.array(data["xyzStop"])
    
    dimXYZ = np.array(data["dimXYZ"])

    voxelSize_DetIn_um = np.array(data["voxelSize_DetIn_um"])
    voxelSize_DetOut_um = np.array(data["voxelSize_DetOut_um"])
       
    cellRadiusMin_um = np.array(data["cellRadiusMin_um"])
    cellRadiusMax_um = np.array(data["cellRadiusMax_um"])
     
    nScales = np.array(data["nScales"])
    
    computingCubeSize = np.array(data["computingCubeSize"])
    computingCubeOverlap = np.array(data["computingCubeOverlap"])
 
    computeTensor = np.array(data["computeTensor"])
    
    nProcess = np.array(data["nProcess"])
    
    args = [pathFolder_ReadImage, pathFolder_WriteResults, 
            xyzStart, xyzStop,
            dimXYZ,
            voxelSize_DetIn_um, voxelSize_DetOut_um,
            cellRadiusMin_um, cellRadiusMax_um,
            nScales,
            computingCubeSize, computingCubeOverlap,
            computeTensor,
            nProcess]
    
    return args

    
# =============================================================================
# Filter
# =============================================================================
def save_jsonFilter(*args):
    
    pathFolder_ReadImage, pathFolder_WriteResults = args[0], args[1]
    Rmin, Rmax = args[2], args[3]
    Imin, Imax = args[4], args[5]

    myDic = {
            "pathFolder_ReadImage"      :     str(Path(str(pathFolder_ReadImage))),
            "pathFolder_WriteResults"   :     str(Path(str(pathFolder_WriteResults))),
            
            "Rmin"  : Rmin.tolist(),
            "Rmax"  : Rmax.tolist(),
                         
            "Imin"  : Imin.tolist(),
            "Imax"  : Imax.tolist(),
              
            }      
        
    #Set Path File to save the json
    rootPath = Path(pathFolder_WriteResults)
    folderName = "Settings"
    fileName   = "Filter.json" 
    pathFolder = Path.joinpath(rootPath, folderName)
    createFolder(str(pathFolder), remove=False)
    pathFile   = Path.joinpath(rootPath, folderName, fileName)
    
    print()
    print('Saving JSON: Filter Settings')
    print(myDic)
    
    with open(pathFile, "w") as write_file: 
        # json.dump(myDic, write_file)
        json.dump(myDic, write_file, indent=1, separators=(", ", ": "), sort_keys=False)  

# =============================================================================
#      
# =============================================================================
if __name__== '__main__':
    import numpy as np
    
    #Initizalization
    pathFolder_ReadImage = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Mini'
    pathFolder_WriteResults = r'C:\Users\aarias\MySpyderProjects\p6_Cell_v10\Results'
    
    voxelSize_In_um  = np.asarray([0.45, 0.45, 2.00])
    voxelSize_Out_um = 2*np.asarray([1.0, 1.0, 1.0])
        

    
    
# =============================================================================
#     
# =============================================================================
    # Create a Python Dictionary Object
    myDic = {
               "pathFolder_ReadImage":     str(Path(pathFolder_ReadImage)),
               "pathFolder_WriteResults":    str(Path(pathFolder_WriteResults)),
               
                "voxelSize_In_um"  : list(voxelSize_In_um),
                "voxelSize_Out_um" : list(voxelSize_Out_um),
      
              #  "xyz_index": list(xyz_index),
              #  "channel": channel,
              #  "ending":  ending,               
              
              #  "xyz_flip": [bool(xyz_flip[0]), bool(xyz_flip[1]), bool(xyz_flip[2])],
              #  "xyz_Dim": list([nx, ny, nz]),
              #  "xyz_Range": list([x0, y0, z0, dx, dy, dz]),
              
              #  "img_Dim": list([img_nx, img_ny, img_nz]),
              #  "img_BitDepth": img_BitDepth,
              #  "img_Format": img_Format,
              
              #  "cameraPixelSize": cameraPixelSize,
              #  "opticalMagnification": opticalMagnification,
              #  "scanOverlap": scanOverlap,
              #  "scan_dx": scan_dx,
              #  "scan_dy": scan_dy,
              
              #  "imgProcessing": imgProcessing,
              #  "isParallel": isParallel,
              #  "nProcesses": nProcesses,
              #  "nThreads": nThreads,
              }      
    
    #Set Path File to save the json
    rootPath = Path(pathFolder_WriteResults)
    folderName = "Settings"
    fileName   = "GUI_Settings.json" 
    pathFolder = Path.joinpath(rootPath, folderName)
    createFolder(str(pathFolder), remove=False)
    pathFile   = Path.joinpath(rootPath, folderName, fileName)
    
    with open(pathFile, "w") as write_file: 
         # json.dump(myDic, write_file)
         json_data = json.dump(myDic, write_file, indent=1, separators=(", ", ": "), sort_keys=False)
    
    print()
    print(json_data)
    print('end') 
# =============================================================================
#     
# =============================================================================
    # dic = {}
    # dic['key'] = 'value'
    # json_data = json.dumps(dic)
    
    # print(json_data) 
# =============================================================================
#         
# =============================================================================
# # Create a Python Dictionary Object
#     myDic = {
#               "pathFile_In":     str('hola'),
#               # "pathRaw_In":    str(pathRaw_In),
#               # "pathRoot_In":    str(pathRoot_In),
#               # "pathRooth_Out":   str(pathRoot_Out),
      
#               # "xyz_index": list(xyz_index),
#               # "channel": channel,
#               # "ending":  ending,               
              
#               # "xyz_flip": [bool(xyz_flip[0]), bool(xyz_flip[1]), bool(xyz_flip[2])],
#               # "xyz_Dim": list([nx, ny, nz]),
#               # "xyz_Range": list([x0, y0, z0, dx, dy, dz]),
              
#               # "img_Dim": list([img_nx, img_ny, img_nz]),
#               # "img_BitDepth": img_BitDepth,
#               # "img_Format": img_Format,
              
#               # "cameraPixelSize": cameraPixelSize,
#               # "opticalMagnification": opticalMagnification,
#               # "scanOverlap": scanOverlap,
#               # "scan_dx": scan_dx,
#               # "scan_dy": scan_dy,
              
#               # "imgProcessing": imgProcessing,
#               # "isParallel": isParallel,
#               # "nProcesses": nProcesses,
#               # "nThreads": nThreads,
#               }      
    
#     #Set Path File to save the json
#     rootPath = Path(pathFolder_WriteResults)
#     folderName = "Settings"
#     fileName   = "GUI_Settings.json" 
#     pathFolder = Path.joinpath(rootPath, folderName)
#     createFolder(str(pathFolder), remove=False)
#     pathFile   = Path.joinpath(rootPath, folderName, fileName)
    
#     with open(pathFile, "w") as write_file: 
#          # json.dump(myDic, write_file)
#          json.dump(myDic, write_file, indent=1, separators=(", ", ": "), sort_keys=False)
         
#     print('end')     
         