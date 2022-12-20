# -*- coding: utf-8 -*-



#GUI 
from PyQt5 import QtWidgets, QtCore 

#Custom GUI Design
from qt_main_VisPy import Ui_MainWindow 

#GPU Scientific Visualization 
import vispy
from vispy import scene
# from vispy.visuals.transforms import STTransform
from vispy.visuals.transforms import STTransform, MatrixTransform, ChainTransform
import vispy.io as io

#Matrix Computation
import numpy as np

#Path Management
from pathlib import Path

#Table Management
import pandas as pd

#System
import sys
import os

#IO
from IO.Image.ImageReader import read_Image, read_ImagePatch
from IO.Image.ImageWriter import save3Dimage_as3DStack
from IO.Files.FileWriter import save_CSV


import time

#Solve Issues
from multiprocessing import freeze_support


#Image 
import tifffile

import matplotlib.pyplot as plt

from IO.Image.ImageReader import read_ImagePatch, get_ImageInfo
from IO.Image.ImageScanner import (get_scanningDistance, get_scanningLocations)
from IO.Files.FileWriter import save_CSV

from ImageProcessing.ImageResampling import resample_3DImage

from GUI_VisPy.VisPyManager import (embed_visViewBoxIntoQtWidget,
                                    add_VisCamToViewBox,
                                    add_visDisplay,
                                    add_visVolObjectToView,
                                    add_visBoxObjectToView,
                                    add_visCubeObjectToView,
                                    add_visGridObjectToView,        
                                    add_visDotsObjectToView,
                                    add_visLineObjectToView,
                                    
                                    update_VisCam,
                                    update_visVol,
                                    
                                    plot_visBox,
                                    plot_visXYZAxis, 
                                    plot_visDots,
                                    # plot_visOrthogonal,
                                    
                                    
                                    update_visDisplay,
                                    update_visOrthoView,
                                    update_visBox, 
                                    update_visCube,
                                    update_visGrid,
                                    update_visPlane,
                                    update_visDots,
                                    update_visLine,
                                    
                                    remove_visObjects,
                                    # add_vis2DDisplay,
                                    # update_vis2DDisplay,
)

from MetaData.JSON_Manager import (save_jsonVisualize, read_jsonVisualize,
                                   save_jsonDetect, read_jsonDetect)

from GUI_QT.QT_Manager import (open_QTDialogToGetpathFolderfromFile, 
                               open_QTDialogToGetpathFolderfromFolder,
                               Worker,
                               PandasModel,
                                ScrollBarPaired
                               )

from IO.Files.FileManager import createFolder


from ImageProcessing.PoseEstimator import run_scanDetections, get_scanningCoordinates

# Globar Varibale

vispy.use('pyqt5')
# =============================================================================
# 
# =============================================================================

#GUI Class
class mywindow(QtWidgets.QMainWindow): 
    def __init__(self): 
        super(mywindow, self).__init__()     
        self.ui = Ui_MainWindow()        
        self.ui.setupUi(self)
        
        #Init GUI-PyQT5 
        self.init_GUI_Events()
        self.init_GUI_Var()
        
        # Init: GUI-VisPy 
        self.init_VisCanvas()
        self.show()
        
        
        # Simulate: users Clicks 
        self.byPass_GUI_Var()
        # self.init_JSON()        
        
    # =============================================================================
    # Initializations
    # =============================================================================
    def init_GUI_Events(self):
        
        # Title
        self.setWindowTitle('Massive MultiScale Cell Detector')
        
        # Orthogonal View
        self.ui.horizontalScrollBar_XY.valueChanged.connect(self.event_ScrollBar_changeSlideXY)
        self.ui.horizontalScrollBar_YZ.valueChanged.connect(self.event_ScrollBar_changeSlideYZ)
        self.ui.horizontalScrollBar_XZ.valueChanged.connect(self.event_ScrollBar_changeSlideXZ)
        
        self.visViewXY_visLineH = add_visLineObjectToView()
        self.visViewXY_visLineV = add_visLineObjectToView()
        
        # Control Orthogonal View
        # m = ScrollBarPaired(self.ui.horizontalScrollBar_Image_Imin, self.ui.horizontalScrollBar_Image_Imax)
        # self.ui.horizontalScrollBar_Image_Imin.valueChanged.connect(self.event_ScrollBar_changeSlideXY)
        # self.ui.horizontalScrollBar_Image_Imin.valueChanged.connect(self.event_ScrollBar_changeSlideXZ)
        
        #Visualize (ReSize: DownSampling)
        self.ui.pushButton_pathFolderReadImage.clicked.connect(self.event_pushButton_pathFolderReadImage) 
        self.ui.pushButton_pathFolderWriteResults.clicked.connect(self.event_pushButton_pathFolderWriteResults) 
        self.ui.pushButton_checkVisIn.clicked.connect(self.event_pushButton_checkVisIn)
        self.ui.pushButton_checkVisOut.clicked.connect(self.event_pushButton_checkVisOut)        
        
        self.ui.pushButton_saveSettingsVisualize.clicked.connect(self.event_pushButton_saveSettingsVisualize)       
        self.ui.pushButton_computeResampledImage.clicked.connect(self.event_pushButton_computeResampledImage)
        self.ui.pushButton_displayResampledImage.clicked.connect(self.event_pushButton_displayResampledImage)
        
        # Select (Crop ROI)
        self.ui.pushButton_cropImage.clicked.connect(self.event_pushButton_cropReSampledImage)
        self.ui.pushButton_uncropImage.clicked.connect(self.event_pushButton_uncropReSampledImage)
        
        # Detect
        self.ui.radioButton_TensorOn.toggled.connect(self.event_radioButton_Tensor)
        self.ui.radioButton_TensorOff.toggled.connect(self.event_radioButton_Tensor)
        
        self.ui.pushButton_checkDet.clicked.connect(self.event_pushButton_checkDet)        
        self.ui.pushButton_saveSettingsDetect.clicked.connect(self.event_pushButton_saveSettingsDetect)
        self.ui.pushButton_computeDetections.clicked.connect(self.event_pushButton_computeDetections)
        self.ui.pushButton_displayDetections.clicked.connect(self.event_pushButton_displayDetections)
        
        # Filter
        self.ui.pushButton_checkFilters.clicked.connect(self.event_pushButton_checkFilters)
        
        self.ui.horizontalScrollBar_Rmin.valueChanged.connect(self.event_ScrollBar_changeRmin)
        self.ui.horizontalScrollBar_Rmax.valueChanged.connect(self.event_ScrollBar_changeRmax)
        
        self.ui.horizontalScrollBar_Nmin.valueChanged.connect(self.event_ScrollBar_change_Nmin)
        self.ui.horizontalScrollBar_Nmax.valueChanged.connect(self.event_ScrollBar_change_Nmax)
        
        self.ui.horizontalScrollBar_Imin.valueChanged.connect(self.event_ScrollBar_changeImin)
        self.ui.horizontalScrollBar_Imax.valueChanged.connect(self.event_ScrollBar_changeImax)
       
        self.ui.horizontalScrollBar_Gmin.valueChanged.connect(self.event_ScrollBar_changeGmin)
        self.ui.horizontalScrollBar_Gmax.valueChanged.connect(self.event_ScrollBar_changeGmax)
        
        self.ui.pushButton_applyFilters.clicked.connect(self.event_pushButton_applyFilters)
        
        # Validate
        self.ui.pushButton_displayTable.clicked.connect(self.pushButton_displayTable)      
        self.ui.tableView.clicked.connect(self.event_clickTable)
        
        # =============================================================================
        #         
        # =============================================================================
        self.compute=False
        self.img3D_dimXYZ = np.zeros(3)

    # =============================================================================
    # Table Management
    # =============================================================================
    def event_clickTable(self, ev):
        
        print('ev.row()', ev.row())
        print('ev.column()', ev.column())
        
        df = self.read_ScanDetectionsFilter()
        df = df.iloc[ev.row()]
        xyz = np.array(df[['X_abs_out_px', 'Y_abs_out_px', 'Z_abs_out_px']]).astype(int)
        
        print(xyz)
        x, y, z = xyz
        
        # visView = self.visView_XYZ
        visView = self.visView_XY
        visDots = self.selected_visDot        
        update_visDots(visView, visDots, xyz, c=[0, 0, 1, 1]) 
        from vispy.visuals.transforms import STTransform
        visDots.transform = STTransform(translate=(0, 0, -200)) 
        visView.add(visDots)
        
        # Get 
        # self.event_ScrollBar_changeSlideXY(z)
        self.set_zPlaneValues(xyz)   
        
        img3D_dimXYZ = self.img3D_dimXYZ

        # XY plane
        visView = self.visView_XY 
        # visView = self.visView_YZ 
        
        #H
        visLine = self.visViewXY_visLineH
        p0 = [0, y]
        p1 = [2000, y]
        p1 = [img3D_dimXYZ[1], y]         
        vertex = np.array([p0, p1])
        update_visLine(visView, visLine, vertex)
        
        #V        
        visLine = self.visViewXY_visLineV
        p0 = [x, 0]
        p1 = [x, 2000]
        p1 = [x, img3D_dimXYZ[0]] 
        vertex = np.array([p0, p1])
        update_visLine(visView, visLine, vertex)

        # # YZ plane
        # visView = self.visView_YZ 
        
        # #H
        # visLine = self.visViewXY_visLineH
        # p0 = [0, z]
        # p1 = [2000, y]
        # p1 = [img3D_dimXYZ[1], z]         
        # vertex = np.array([p0, p1])
        # update_visLine(visView, visLine, vertex)
        
        # #V        
        # visLine = self.visViewXY_visLineV
        # p0 = [y, 0]
        # p1 = [y, 2000]
        # p1 = [y, img3D_dimXYZ[2]] 
        # vertex = np.array([p0, p1])
        # update_visLine(visView, visLine, vertex)
        
# =============================================================================
#         
# =============================================================================
    def init_GUI_Var(self):
        #Load
        self.pathFolder_ReadImage    = None
        self.pathFolder_WriteResults = None
        self.voxelSize_VisIn_um  = np.array([None, None, None])
        self.voxelSize_VisOut_um = np.array([None, None, None])
        
        #Images
        self.img3D = None
        self.img2D_XY = None
        self.img2D_YZ = None
        self.img2D_XZ = None
        
        
        # Orthogonal View
        self.visViewPoint = scene.visuals.Markers()
        
        #ROI
        # self.selectStart = scene.visuals.Markers()
        self.visSelectPoint = scene.visuals.Markers()
        self.visSelectBox  = scene.visuals.Box(width=1, height=1, depth=1, 
                                               color=(0, 0, 1, 0.001),
                                               edge_color=(1, 0, 1, 0.9),
                                               )
        # self.visSelectBox.set_gl_state('translucent', depth_test=False) 
        
        #Detect Display
        self.visDots_XYZ = add_visDotsObjectToView()
        self.visCubes = []
        self.biestableClick = True
        
        #Selected Points
        self.selected_visDot = add_visDotsObjectToView()
  
    def init_VisCanvas(self):
        # Forcing the Backend PyQt5
        # vispy.use('pyqt5')
        # print()
        # print('vispy.sys_info(): \n', vispy.sys_info())
        
        # =============================================================================
        #         
        # =============================================================================

        # Init visViews     
        # add_visDisplay(self.ui.visWidget_3D, 'Turntable')              
        visView, visCam, visVol, visBox, visAxes = add_visDisplay(self.ui.visWidget_3D, 'Turntable')
        self.visView_XYZ, self.visCam_XYZ, self.visVol_XYZ, self.visBox_XYZ, self.visAxes_XYZ = visView, visCam, visVol, visBox, visAxes
        self.visOrtho_XYZ = visView, visCam, visVol, visBox, visAxes
        
        visView, visCam, visVol, visBox, visAxes = add_visDisplay(self.ui.visWidget_XY, 'PanZoom')
        self.visView_XY, self.visCam_XY, self.visVol_XY, self.visBox_XY, self.visAxes_XY = visView, visCam, visVol, visBox, visAxes
        self.visOrtho_XY = visView, visCam, visVol, visBox, visAxes

        visView, visCam, visVol, visBox, visAxes = add_visDisplay(self.ui.visWidget_YZ, 'PanZoom')
        self.visView_YZ, self.visCam_YZ, self.visVol_YZ, self.visBox_YZ, self.visAxes_YZ = visView, visCam, visVol, visBox, visAxes
        self.visOrtho_YZ = visView, visCam, visVol, visBox, visAxes
        
        visView, visCam, visVol, visBox, visAxes = add_visDisplay(self.ui.visWidget_XZ, 'PanZoom')
        self.visView_XZ, self.visCam_XZ, self.visVol_XZ, self.visBox_XZ, self.visAxes_XZ = visView, visCam, visVol, visBox, visAxes
        self.visOrtho_XZ = visView, visCam, visVol, visBox, visAxes
             
        #Init Volumes
        img3D = np.ones((100, 100, 100))
        visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ = self.get_visOrthoView()
        update_visOrthoView(img3D, visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ)
        
        # Init visEvents
        self.visView_XY.events.mouse_press.connect(self.on_mouse_press_XY)
        self.visView_XYZ.events.mouse_press.connect(self.on_mouse_press_XYZ)
        
        
        
        
        
        #Init: visEvents
        # self.visView_XY.events.mouse_press.connect(self.on_mouse_press0)
        # self.canvasXY.events.mouse_press.connect(self.on_mouse_press0)
        # self.canvasXY.events.key_press.connect(self.on_key_press)
               
        # self.visViewBox_XY.events.mouse_press.connect(self.on_mouse_press)      
        # self.visViewBox_XY.camera.viewbox_mouse_event(self.on_mouse_press)
        # self.visViewBox_XYZ.events.mouse_press.connect(self.on_mouse_press)
        
        #Dots in XYZ visView
        # visView_XYZ = self.visView_XYZ
        


# =============================================================================
#  Simulation of User Clicks       
# =============================================================================
    def byPass_GUI_Var(self):

        # =============================================================================
        # Visualize        
        # =============================================================================
        # Select: ReadPath
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\ImageWholeBrainx32' #Whole
        pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\Ref_1900_13678_15x_full\ParaStitched\RES(580x450x230)' #Whole
        pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\Ref_1900_13678_15x_Hipo\Ref_1900_13678_Raw_Stitched\RES(430x431x78)'
        pathFolder_ReadImage    = 'F:\Arias\BrainTempProcessing\Ref_1900_13678_15x_Hipo\Ref_1900_13678_Raw_Stitched\Vis_AxonBundles'
        
        pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\Ref_1900_13678_15x_full\Ref_1900_13678_Raw_TeraStitcher\067892\067892_022630' #Axon bundles
        
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\ImageTile' #Dendrites
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\ImageTileMini'
        # pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\Ref_2021_36280_Tile_15x_tif'
        
        pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\MiniTest' # High Cell Density
        
        pathFolder_ReadImage    = Path().absolute() / "Examples\MiniTest"
        # pathFolder_ReadImage    = r'F:\Arias\MyTemp\03_Stitched\RES(7076x7076x40)'
        
        # pathFolder_ReadImage    = r'F:\Arias\MyTemp\03_Stitched\RES(7076x7076x40)'
        # pathFolder_ReadImage    =  r'F:\Arias\BrainTempProcessing\Ref_Alvaro_1_Hippo_Test_594_LP607_tile_x4_y3\3D_100'
         
        
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Miguel\DataSet_2\PV'
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Miguel\DataSet_2\FOS'
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Miguel\DataSet_2\FOS_crop'
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Miguel\DataSet_2\FOS_pad'
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Miguel\DataSet_2\FOS_crop_pad'
        # pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\New folder\Ref_1900_13678_4_4x_full_TeraStitcher\000000\000000_007543'

        # pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\Ref_1900_13678_15x_Hipo\Ref_1900_13678_Raw_Stitched\RES(3441x3449x631)'

        # pathFolder_ReadImage    = r'F:\Arias\BrainTempProcessing\Ref_1900_13678_15x_Hipo\Ref_1900_13678_Raw_Stitched\RES(3441x3449x631)'
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Miguel\DataSet_3\PV'
        
        #Paulina
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\ImageDataSets\Paulina\LSM980\Control'
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\ImageDataSets\Paulina\LSM980\Control_pad'
        # pathFolder_ReadImage    = r'C:\Users\aarias\MyPipeLine\ImageDataSets\Paulina\LSM980\TH_pad'

    
        #Whole Brain
        # pathFolder_ReadImage      = r'F:\Arias\BrainTempProcessing\Ref_1900_13678_4_4x_full\Stitched_Uncompressed'
        
        
        self.pathFolder_ReadImage    = pathFolder_ReadImage
        
        # Select: WritePath
        # pathFolder_WriteResults = r'C:\Users\aarias\MySpyderProjects\p6_Cell_v11\Results\mainTest'
        localPath = os.path.dirname(sys.argv[0])
        rootName = 'Results'
        BrainRegion = 'mainTest'
        pathFolder_WriteResults = os.path.join(localPath, rootName, BrainRegion) 
        self.pathFolder_WriteResults = pathFolder_WriteResults
        
        #Create Folder
        createFolder(str(Path(pathFolder_WriteResults)), remove=False)
        
        #PushButton: ImgIn
        self.event_pushButton_checkVisIn()
        # jaja
        
        # Resampling:
        voxelSize_VisIn_um  = np.asarray([0.45, 0.45, 2.00])
        voxelSize_VisOut_um = 2*np.asarray([1.0, 1.0, 1.0]) 
        # voxelSize_VisOut_um = 1*np.asarray([0.45, 0.45, 2.00])
        
        if pathFolder_ReadImage == r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\ImageWholeBrainx32':
            voxelSize_VisIn_um  = 1*np.asarray([1.0, 1.0, 1.0])
            voxelSize_VisOut_um = 1*np.asarray([1.0, 1.0, 1.0])
        
        if pathFolder_ReadImage ==  r'F:\Arias\BrainTempProcessing\Ref_1900_13678_15x_full\ParaStitched\RES(580x450x230)':
            voxelSize_VisIn_um  = 1*np.asarray([1.0, 1.0, 1.0])
            voxelSize_VisOut_um = 1*np.asarray([1.0, 1.0, 1.0])
        
        if pathFolder_ReadImage == r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Miguel\DataSet_3\PV':
            voxelSize_VisIn_um  = 1*np.asarray([0.76, 0.76, 3.0])
            voxelSize_VisOut_um = 2*np.asarray([1.0, 1.0, 1.0])
            # voxelSize_VisOut_um = 15*np.asarray([1.0, 1.0, 1.0])
        
        if pathFolder_ReadImage == r'F:\Arias\BrainTempProcessing\New folder\Ref_1900_13678_4_4x_full_TeraStitcher\000000\000000_007543':
            voxelSize_VisIn_um  = 1*np.asarray([1.48, 1.48, 2.0])
            voxelSize_VisOut_um = 2*np.asarray([1.0, 1.0, 1.0])
            
        
        if pathFolder_ReadImage == r'F:\Arias\BrainTempProcessing\Ref_1900_13678_4_4x_full\Stitched_Uncompressed':
            voxelSize_VisIn_um  = 1*np.asarray([1.0, 1.0, 1.0])
            voxelSize_VisOut_um = 1*np.asarray([1.0, 1.0, 1.0])
        
        if pathFolder_ReadImage == r'F:\Arias\BrainTempProcessing\Ref_1900_13678_15x_Hipo\Ref_1900_13678_Raw_Stitched\RES(3441x3449x631)':
            voxelSize_VisIn_um  = 1.8*np.asarray([1.0, 1.0, 1.0])
            voxelSize_VisOut_um = 1.8*np.asarray([1.0, 1.0, 1.0])
        
        #Paulina: Voxel Resolution (0.59, 0.59, 3um)
        if  'Paulina' in str(pathFolder_ReadImage):
            voxelSize_VisIn_um  = 1*np.asarray([0.59, 0.59, 3.0])
            voxelSize_VisOut_um = 2*np.asarray([1.0, 1.0, 1.0]) 

        self.set_ReSamplingVis(voxelSize_VisIn_um, voxelSize_VisOut_um)
        
        #PushButton: ImgOut
        self.event_pushButton_checkVisOut()
        
        #PushButton: Save Settings
        self.event_pushButton_saveSettingsVisualize()
        
        #toBypassthe Resampling Computation
        # self.event_pushButton_displayResampledImage() #toBypassthe Resampling Computation
        
        # # PushButton: Compute ReSampling
        # #ByPass
        # isResEqual = (voxelSize_VisIn_um==voxelSize_VisOut_um).all()
        # print()
        # print('isResEqual=', isResEqual)
        # if isResEqual:
        #     self.isResampled = False
        # else:
        #     self.isResampled = True
        #     self.compute = True
        #     self.event_pushButton_computeResampledImage()
        
        # # while self.compute==True:
        #     # time.sleep(2)
        # print()
        # print('isResampled=', self.isResampled)
        # # self.event_pushButton_displayResampledImage() #???
        
        
        # =============================================================================
        # Selection    
        # =============================================================================
        
        #Set Start
        xyz_start = np.array([0,0,0])
        self.set_selectStartXYZ(xyz_start)
        
        dimXYZ, bitDepth, fileExtension, memSize = self.get_checkVisIn()
        xyz_stop = np.array(dimXYZ-1)
        self.set_selectStopXYZ(xyz_stop)
        
        # =============================================================================
        # Detection
        # =============================================================================

        # Resampling:
        voxelSize_DetIn_um  = np.asarray([0.45, 0.45, 2.00])
        voxelSize_DetOut_um = 2*np.asarray([1.0, 1.0, 1.0]) 
        
        if pathFolder_ReadImage == r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Miguel\DataSet_3\PV':
            voxelSize_DetIn_um  = 1*np.asarray([0.76, 0.76, 3.0])
            voxelSize_DetOut_um = 2*np.asarray([1.0, 1.0, 1.0])
       
        if pathFolder_ReadImage == r'F:\Arias\BrainTempProcessing\New folder\Ref_1900_13678_4_4x_full_TeraStitcher\000000\000000_007543':
            voxelSize_DetIn_um  = 1*np.asarray([1.48, 1.48, 2.0])
            voxelSize_DetOut_um = 2*np.asarray([1.0, 1.0, 1.0])

        #Paulina: Voxel Resolution (0.59, 0.59, 3um)
        if  'Paulina' in str(pathFolder_ReadImage):
            voxelSize_DetIn_um  = 1*np.asarray([0.59, 0.59, 3.0])
            voxelSize_DetOut_um = 2*np.asarray([1.0, 1.0, 1.0]) 

            
        self.set_ReSamplingDet(voxelSize_DetIn_um, voxelSize_DetOut_um)

        # Radius:
        cellRadiusMin_um  = np.asarray([5])
        cellRadiusMax_um  = np.asarray([15])
        self.set_cellRadiusRange(cellRadiusMin_um, cellRadiusMax_um)        

        # nScales:
        nScales  = np.asarray([8])
        self.set_nScales(nScales)  
        

        # Computing CubeSize
        computingCubeSize = np.asarray([2])
        computingCubeSize = np.asarray([6])
        computingCubeSize = np.asarray([3])
        self.set_computingCubeSize(computingCubeSize)

        # Computing Overlap
        computingCubeOverlap = np.asarray([2])
        computingCubeOverlap = np.asarray([0])
        self.set_computingCubeOverlap(computingCubeOverlap)  
        
        # Tensor Bool
        computeTensor = False
        self.set_computeTensor(computeTensor)
        
        # nProcess 
        nProcess = np.asarray([2])
        nProcess = np.asarray([8])
        self.set_nProcess(nProcess) 
        
        # =============================================================================
        # Resample, Detect and Display Detection  
        # =============================================================================
        
        #ByPass
        isResEqual = (voxelSize_VisIn_um==voxelSize_VisOut_um).all()
        print()
        print('isResEqual=', isResEqual)
        if isResEqual:
            self.isResampled = False
        else:
            self.isResampled = True
            self.event_pushButton_computeResampledImage()
        
        print()
        print('isResampled=', self.isResampled)
           

        
    def init_JSON(self):
        pass





# =============================================================================
#         Cosas Utiles
# =============================================================================

        # print('visView.camera.get_state()', visView.camera.get_state())# ????            
        # visView.camera.set_state() #????
        
# =============================================================================
#  Handlers of VisPy Events
# =============================================================================
    def on_key_press(self, ev):
        print()
        print('on_key_press')
        print('ev', ev)
        
    def on_mouse_press_0(self, ev):
        print()
        print('on_mouse_press')        
        tab_ix = self.ui.tabWidget.currentIndex()        
        print('Current Tab:', tab_ix)
        print('ev.button:', ev.button) 
    
    def on_mouse_press_XYZ(self, ev):
        print()
        print('on_mouse_press')        
        tab_ix = self.ui.tabWidget.currentIndex()        
        print('Current Tab:', tab_ix)
        print('ev.button:', ev.button)
        if (ev.button==1):
            view = self.visView_XY
            
            # Get the Out/In Ratio 
            voxelSize_VisIn_um, voxelSize_VisOut_um = self.get_ReSamplingVis()
            r_um = voxelSize_VisOut_um/voxelSize_VisIn_um
            r_px = 1/r_um
            
            #Transform to the World Coordinates
            tf = view.scene.transform
            xyz_Resampled = tf.imap(ev.pos)
            xyz_Resampled = np.round(xyz_Resampled[0:3]).astype(np.uint32)
            
            #Transform relative to Resampled Image
            xyz_Original = r_um*xyz_Resampled
            xyz_Original = (np.round(xyz_Original)).astype(np.uint32) # ??? +2
            print()
            print('xyz_Resampled:', xyz_Resampled)  
            print('xyz_Original:', xyz_Original)             
    
    # def on_mouse_press_XYZ(self, ev):
    #     view = self.visView_XY
        
    #     tform=view.scene.transform
    #     w,h = view.canvas.size
    #     screen_center = np.array([w/2,h/2,0,1]) # in homogeneous screen coordinates
    #     d1 = np.array([0,0,1,0]) # in homogeneous screen coordinates
    #     point_in_front_of_screen_center = screen_center + d1 # in homogeneous screen coordinates
        
    #     point_in_front_of_screen_center = ev.pos
        
    #     p1 = tform.imap(point_in_front_of_screen_center) # in homogeneous scene coordinates
    #     p0 = tform.imap(screen_center) # in homogeneous screen coordinates
    #     assert(abs(p1[3]-1.0) < 1e-5) # normalization necessary before subtraction
    #     assert(abs(p0[3]-1.0) < 1e-5)
        
    #     print()
    #     print(p0)
    #     print(p1)
        
    #     return p0[0:3],p1[0:3] # 2 point representation of view axis in 3d scene coordinates  
      
    def on_mouse_press_XY(self, ev):
        print()
        print('on_mouse_press')        
        tab_ix = self.ui.tabWidget.currentIndex()        
        print('Current Tab:', tab_ix)
        print('ev.button:', ev.button)
        
        
        
        if (ev.button==1) & (tab_ix==2):
            view = self.visView_XY  
            biestableClick = self.biestableClick
            visSelectPoint = self.visSelectPoint
            visSelectBox = self.visSelectBox
            
            # Get the Out/In Ratio 
            voxelSize_VisIn_um, voxelSize_VisOut_um = self.get_ReSamplingVis()
            r_um = voxelSize_VisOut_um/voxelSize_VisIn_um
            r_px = 1/r_um
            
            #Transform to the World Coordinates
            tf = view.scene.transform
            xyz_Resampled = tf.imap(ev.pos)
            xyz_Resampled = np.round(xyz_Resampled[0:3]).astype(np.uint32)
            
            #Transform relative to Resampled Image
            xyz_Original = r_um*xyz_Resampled
            xyz_Original = (np.round(xyz_Original)).astype(np.uint32) # ??? +2
            print()
            print('xyz_Resampled:', xyz_Resampled)  
            print('xyz_Original:', xyz_Original)  
            
            # dimXYZ, bitDepth, fileExtension, memSize = self.get_checkVisOut()
            # dimZYX = dimXYZ[[2, 1, 0]]
            # update_visBox(visView=view, visBox=visSelectCropRect, dimZYX=dimZYX)
            
            # v_p = np.array([[100, 100, 0]])
            # v_p = r_px*v_p
            # v_p = np.array([xyz_Resampled])
            # visDots = plot_visDots(view, v_p, R=10)
                  
            # xy = np.array([xyz_Resampled[0], xyz_Resampled[1]])
            # xy_norm = np.array([xyz_Original[0], xyz_Original[1]]) 
            if biestableClick==True:
                biestableClick = False
                
                #Add Rectangle
                view.add(visSelectPoint)
                visSelectPoint.set_data(np.array([xyz_Resampled]),
                                            face_color = [0, 1, 0, 0.5],
                                            size=5
                                              )                
                self.set_selectStartXYZ(xyz_Original)
                
            else:                
                biestableClick = True
                
                #Remove Point
                view.add(visSelectPoint)
                visSelectPoint.parent = None                
                self.set_selectStopXYZ(xyz_Original)
                
                
                xyzStart, xyzStop = self.get_selectCropXYZ()
                xyzStart = (np.round(r_px*xyzStart)).astype(np.uint32)
                xyzStop = (np.round(r_px*xyzStop)).astype(np.uint32)
                dimXYZ = xyzStop - xyzStart
                
                
                view.add(visSelectBox)
                tx = xyzStart[0] + dimXYZ[0]//2
                ty = xyzStart[1] + dimXYZ[1]//2
                tz = xyzStart[2] + dimXYZ[2]//2
                t = np.array([tx, ty, tz])
                
                sx = dimXYZ[0]
                sy = dimXYZ[1]
                sz = dimXYZ[2]
                s = np.array([sx,sy, sz])
                print()
                print('hello')
                print(t)
                print(s)
                visSelectBox.transform = STTransform(translate=(tx, ty, tz),
                                                                      scale=(sx, sy, sz))
                visSelectBox.order = 2
                # visSelectCropRect(width=20, height=20, depth=20)
                
                # self.add_visRectangle(view, visSelectCropRect, xyStart, xyStop)

                # visSelectCropRect.transform.scale = sc
              
            self.biestableClick = biestableClick    
           
            print()
            print('biestableClick:', self.biestableClick)                    
 
             
# =============================================================================
#    GUI Event Handlers  
# =============================================================================

    # =============================================================================
    #     Events: Visualize          
    # =============================================================================
    def event_pushButton_pathFolderReadImage(self): 
        print ('')
        print('Start: event_select_pathFolderReadImage()')
        
        pathFolder = open_QTDialogToGetpathFolderfromFile(self)                                
        self.pathFolder_ReadImage = str(pathFolder)
        
        print ('')
        print('Stop : event_select_pathFolderReadImage()')
   
    def event_pushButton_pathFolderWriteResults(self):
        print ('')
        print('Start: event_pushButton_pathFolderWriteResults()')

        pathFolder = open_QTDialogToGetpathFolderfromFolder(self)
        self.pathFolder_WriteResults = str(pathFolder)
        
        print ('')
        print('Stop : event_pushButton_pathFolderWriteResults()')
        
    def event_pushButton_checkVisIn(self):
        print('')
        print('Start: event_getImageInputInfo()')        
        pathFolder_ReadImage = self.pathFolder_ReadImage
        imgDimXYZ, bitDepth, fileExtension, memSize = get_ImageInfo(pathFolder_ReadImage)
        self.set_checkVisIn(imgDimXYZ, bitDepth, fileExtension, memSize)
        print('')
        print('Stop : event_getImageInputInfo()')


    def event_pushButton_checkVisOut(self):
        print('')
        print('Start: event_pushButton_checkVisOut()')
        
        # Get ImgIn Information
        dimXYZ_VisIn, bitDepth, fileExtension, memSize_VisIn = self.get_checkVisIn()
        
        # Get Resampling Settings
        voxelSize_VisIn_um, voxelSize_VisOut_um =  self.get_ReSamplingVis()
                
        # ??? Update VoxelSize        
        self.voxelSize_VisIn_um = voxelSize_VisIn_um 
        self.voxelSize_VisOut_um = voxelSize_VisOut_um 
        
        # Get the Out/In Ratio 
        r_um = voxelSize_VisOut_um/voxelSize_VisIn_um
        r_px = 1/r_um
        
        # Predict ImgOut Dimension
        imgDimXYZ_Out = (np.round(r_px*dimXYZ_VisIn)).astype(int)
        
        # Predict ImgOut MemorySize in GigaBytes
        memSize_VisOut = (bitDepth/8)*(np.prod(imgDimXYZ_Out.astype(float))/10**9) 
        
        # Show the Predicted ImgOut Information
        self.set_checkVisOut(imgDimXYZ_Out, bitDepth, fileExtension, memSize_VisOut)
     
        print('')
        print('Stop : event_pushButton_checkVisOut()')
        
    def event_pushButton_saveSettingsVisualize(self):
        print('')
        print('Start: event_pushButton_saveSettingsVisualize()')     
        
        self.save_settingsVisualize()
        
        print('')
        print('Stop : event_pushButton_saveSettingsVisualize()')
    
    def event_pushButton_computeResampledImage(self):
        print('')
        print('Start: event_pushButton_computeResampledImage()')     
        
        # Compute using a New Thread
        self.work_onComputeResampledImage()  
        
        # Compute using a the UI Thread (freeze the UI)
        # self.computeResampledImage() ·#???? Warning: freeze the UI
        
        print('')
        print('Stop : event_pushButton_computeResampledImage()')
        
    def event_pushButton_displayResampledImage(self):
        print('')
        print('Start: event_pushButton_displayResampledImage()')
        
        print('------------------------')
        print('----------44444444444444------------')
        print('------------------------')
        
        # img3D = self.get_img3D()
        # if img3D == None:
        img3D = self.read_resampledImage(self.pathFolder_WriteResults)
        
        visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ = self.get_visOrthoView()
        update_visOrthoView(img3D, visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ)
        
        [ny, nx, nz] = img3D.shape
        dimXYZ = np.array([nx, ny, nz])
        dimXYZ_min = np.zeros(3)
        dimXYZ_max = np.array([nx, ny, nz]) - 1
        self.set_zPlaneRanges(xyz_min=dimXYZ_min, xyz_max=dimXYZ_max)
        self.img3D_dimXYZ = dimXYZ
        xyz = dimXYZ//2
        
        # =============================================================================
        #         
        # =============================================================================
        #dfgd ????
        visView = self.visView_XY
        [x, y, z] = xyz
        
        
        #H
        visLine = self.visViewXY_visLineH
        p0 = [0, y]
        p1 = [2000, y]
        p1 = [dimXYZ[1], y]        
        vertex = np.array([p0, p1])
        update_visLine(visView, visLine, vertex)
        
        #V        
        visLine = self.visViewXY_visLineV
        p0 = [x, 0]
        p1 = [x, 2000]        
        p1 = [x, dimXYZ[0]]
        vertex = np.array([p0, p1])
        update_visLine(visView, visLine, vertex)

        
        self.set_zPlaneValues(xyz)
                
        # =============================================================================
        #         
        # =============================================================================
        
        # self.set_img3D(img3D)
        print('xyz', xyz)
        print('dimXYZ', dimXYZ)
        print('')
        print('Stop : event_pushButton_displayResampledImage()')
        self.event_pushButton_computeDetections()  #??? bypassing
        
    

 
    # =============================================================================
    #   Events: Select ROI      
    # =============================================================================
    def event_pushButton_cropReSampledImage(self):
        print('')
        print('Start: event_pushButton_cropReSampledImage()')
        
        #Read ReSampled (e.g. DownSampled) Image from the project folder (ReSampled)
        img3D = self.read_resampledImage(self.pathFolder_WriteResults)
        # img3D = self.get_img3D()
        
        #Compute Crop Image
        xyzStart, xyzStop = self.get_selectCropXYZ()
        voxelSize_VisIn_um, voxelSize_VisOut_um = self.get_ReSamplingVis()
        r_um = voxelSize_VisOut_um/voxelSize_VisIn_um
        r_px = 1/r_um
            
        xyzStart = (np.round(r_px*xyzStart)).astype(np.uint32)
        xyzStop = (np.round(r_px*xyzStop)).astype(np.uint32)
        img3D_Cropped = self.compute_cropImage(img3D, xyzStart, xyzStop)
        
        print()
        print('xyzStart:', xyzStart)
        print('xyzStop:', xyzStop)
        print('img3D_Cropped.shape:', img3D_Cropped.shape)
        
        # Plot Image in VisPy Orthogonal 
        # visViewBox_XYZ, visViewBox_XY, visViewBox_YZ, visViewBox_XZ = self.get_visViewBox()
        # plot_visOrthogonal(img3D_Cropped, visViewBox_XYZ, visViewBox_XY, visViewBox_YZ, visViewBox_XZ)
        visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ = self.get_visOrthoView()
        update_visOrthoView(img3D_Cropped, visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ)

        print('')
        print('Stop : event_pushButton_cropReSampledImage()')

    def event_pushButton_uncropReSampledImage(self):
        print('')
        print('Start: event_pushButton_uncropReSampledImage()')        
        # self.set_img3D(None)
        self.event_pushButton_displayResampledImage()
        
        print('')
        print('Stop : event_pushButton_uncropReSampledImage()')

    # =============================================================================
    #    Events: Detection
    # =============================================================================
    def event_radioButton_Tensor(self, ev):
        print('')
        print('Start: event_radioButton_Tensor()')
        
        pass
        # radioButton = self.sender()
        # print('radioButton.text()', radioButton.text())
        # if radioButton.text()=='Yes':
        #     pass
        # elif radioButton.text()=='No':
        #     pass
        # else:
        #     print()
        #     print('evet_radioButton_Tensor(): radioButton.text() not found')
            
        print('')
        print('Stop : event_radioButton_Tensor()')  

    def event_pushButton_checkDet(self):
        
        # Save Detection Settings
        self.save_settingsDetect()
        
        # Load Detection Settings
        pathFile = self.get_pathFile_SettingsDetect()
        args = read_jsonDetect(pathFile)
        
        #Unpack Detection Parameters
        pathFolder_ReadImage, pathFolder_WriteResults = args[0], args[1]
        scannerStart_In_px, scannerEnd_In_px = args[2], args[3]
        imgDimXYZ = args[4]
        voxelSize_In_um, voxelSize_Out_um = args[5], args[6]
        cellRadiusMin_um, cellRadiusMax_um = args[7], args[8]
        nScales = args[9]
        computingCubeSize, computingCubeOverlap = args[10], args[11]
        computeTensor = args[12]
        nProcess = args[13]
        
        # OverWrite the Saving Path to store in a SubFolder 
        rootPath = Path(pathFolder_WriteResults)
        folderName = "Detections"
        pathFolder_WriteResults   = Path.joinpath(rootPath, folderName)
        
        #Voxel Ratio: In/Out
        voxelSize_Ratio_um = voxelSize_Out_um/voxelSize_In_um
        voxelSize_Ratio_px = 1.0/voxelSize_Ratio_um
            
        # Computing Sanning Locations
        args = get_scanningCoordinates(scannerStart_In_px, scannerEnd_In_px, imgDimXYZ, voxelSize_In_um, voxelSize_Out_um, cellRadiusMax_um, computingCubeSize, computingCubeOverlap)
        scannerPositions_In_px, scannerSize_In_px, scannerOverlap_In_px = args[0], args[1], args[2]
        scannerPositions_Out_px, scannerSize_Out_px, scannerOverlap_Out_px = args[3], args[4], args[5]
        
        # print()
        # print('scannerPositions_In_px \n', scannerPositions_In_px)
        # print('scannerSize_In_px \n', scannerSize_In_px)
        # print()        
        # print('scannerPositions_Out_px \n', scannerPositions_Out_px)
        # print('scannerSize_Out_px \n', scannerSize_Out_px)
        # print()
        # print('nPositions \n', scannerPositions_In_px.shape)
        cubeSizeStr = ('{:0.0f}'.format(scannerSize_In_px[0]) + 'x' +
                        '{:0.0f}'.format(scannerSize_In_px[1]) + 'x' +
                        '{:0.0f}'.format(scannerSize_In_px[2])
                        )
        self.ui.label_cubeInSize_px.setText(cubeSizeStr)
        
        cubeSizeStr = ('{:0.0f}'.format(scannerSize_Out_px[0]) + 'x' +
                        '{:0.0f}'.format(scannerSize_Out_px[1]) + 'x' +
                        '{:0.0f}'.format(scannerSize_Out_px[2])
                        )
        self.ui.label_cubeOutSize_px.setText(cubeSizeStr)  
        nCubes = '{:0.0f}'.format(scannerPositions_Out_px.shape[0])
        self.ui.label_nCubes.setText(nCubes)
        
        
        remove_visObjects(self.visCubes)
        self.visCubes = []
        
        visView = self.visView_XYZ  
        dimXYZ = scannerSize_Out_px
        k = 0
        for pos in scannerPositions_Out_px: 
            k = k + 1
            visCube = add_visCubeObjectToView(visView)            
            update_visCube(visView, visCube, pos, dimXYZ, mode='center')
            self.visCubes.append(visCube)
            # if k==8:
            #     break
            
        
        
    def event_pushButton_saveSettingsDetect(self):
        print('')
        print('Start: event_pushButton_saveSettingsDetect()')     
        
        self.save_settingsDetect()
        
        print('')
        print('Stop : event_pushButton_saveSettingsDetect()')     

    def event_pushButton_computeDetections(self):
        print('')
        print('Start: event_pushButton_computeDetections()')     
        
        self.ui.pushButton_computeDetections.setEnabled(False)
        self.ui.pushButton_displayDetections.setEnabled(False)
        
        self.work_onComputeScanDetections()
        
        print('')
        print('Stop : event_pushButton_computeDetections()') 
        
    def event_pushButton_displayDetections(self):
        print('')
        print('Start: event_pushButton_displayDetections()') 
        # Get DataFrame with Detections        
        df = self.read_ScanDetections()
        self.display_Detections(df)
        print('')
        print('Stop : event_pushButton_displayDetections()')
        
    # =============================================================================
    #    Events: Filter
    # ============================================================================= 
    
    def event_pushButton_checkFilters(self):
        print('')
        print('Start: event_pushButton_checkFilters()')   
    
        Rmin, Rmax, Nmin, Nmax, Imin, Imax, Gmin, Gmax = self.get_filterExtremes() 
        self.set_filterRanges(Rmin, Rmax, Nmin, Nmax, Imin, Imax, Gmin, Gmax)

        print('')
        print('Stop : event_pushButton_checkFilters()')

    def event_ScrollBar_changeRmin(self, value):
        Rmin = value
        Rmax = self.ui.horizontalScrollBar_Rmax.value()
        if Rmin>Rmax:
            self.ui.horizontalScrollBar_Rmax.setValue(Rmin)            
        self.ui.label_Rmin.setText(str(Rmin))

    def event_ScrollBar_changeRmax(self, value):
        Rmin = self.ui.horizontalScrollBar_Rmin.value() 
        Rmax = value
        if Rmin>Rmax:
            self.ui.horizontalScrollBar_Rmin.setValue(Rmax)            
        self.ui.label_Rmax.setText(str(Rmax))

    def event_ScrollBar_change_Nmin(self, value):
        Nmin = value
        Nmax = self.ui.horizontalScrollBar_Nmax.value()
        if Nmin>Nmax:
            self.ui.horizontalScrollBar_Nmax.setValue(Nmin)            
        self.ui.label_Nmin.setText(str(Nmin))

    def event_ScrollBar_change_Nmax(self, value):
        Nmin = self.ui.horizontalScrollBar_Nmin.value() 
        Nmax = value
        if Nmin>Nmax:
            self.ui.horizontalScrollBar_Nmin.setValue(Nmax)            
        self.ui.label_Nmax.setText(str(Nmax))
        
        
    def event_ScrollBar_changeImin(self, value):
        Imin = value
        Imax = self.ui.horizontalScrollBar_Imax.value()
        if Imin>Imax:
            self.ui.horizontalScrollBar_Imax.setValue(Imin)            
        self.ui.label_Imin.setText(str(Imin))

    def event_ScrollBar_changeImax(self, value):
        Imin = self.ui.horizontalScrollBar_Imin.value() 
        Imax = value
        if Imin>Imax:
            self.ui.horizontalScrollBar_Imin.setValue(Imax)            
        self.ui.label_Imax.setText(str(Imax))

    def event_ScrollBar_changeGmin(self, value):
        Gmin = value
        Gmax = self.ui.horizontalScrollBar_Gmax.value()
        if Gmin>Gmax:
            self.ui.horizontalScrollBar_Gmax.setValue(Gmin) 
        self.ui.label_Gmin.setText(str(Gmin))

    def event_ScrollBar_changeGmax(self, value):
        Gmin = self.ui.horizontalScrollBar_Gmin.value() 
        Gmax = value
        if Gmin>Gmax:
            self.ui.horizontalScrollBar_Gmin.setValue(Gmax)            
        self.ui.label_Gmax.setText(str(Gmax))
        
        
    def event_pushButton_applyFilters(self):
        print('')
        print('Start: event_pushButton_applyFilters()') 
        self.apply_Filters()        
        print('')
        print('Stop: event_pushButton_applyFilters()') 
        pass

    # =============================================================================
    #    Events: Validation 
    # =============================================================================

    def pushButton_displayTable(self):
        df = self.read_ScanDetectionsFilter()
        df = df[['ID', 'S_um','I_Raw','I_Pro','I_DoG','N']]
        df = df.astype(int)
        model = PandasModel(df)
        self.ui.tableView.setModel(model)

    
# =============================================================================
#   Manage Paths
# =============================================================================
    # rootPath = Path(r'C:\Users\aarias\MySpyderProjects\p6_Cell_v13\Results\mainTest')
    
    def get_pathFolder_ReadImage(self):
        return self.pathFolder_ReadImage

    def get_pathFolder_WriteResults(self):
        return self.pathFolder_WriteResults  
    
    def get_pathFolder_ImageResampled(self):        
        rootPath = Path(self.pathFolder_WriteResults)
        folderName = "ImageResampled"
        pathFolder   = Path.joinpath(rootPath, folderName) 
        return pathFolder
    
    def get_pathFile_ImageResampled(self, isResampled=False):
        if isResampled==True:
            rootPath = Path(self.pathFolder_WriteResults)
            folderName = "ImageResampled"
            fileName   = "Visualize.tif"         
            pathFile   = Path.joinpath(rootPath, folderName, fileName)  
            print('-----------------------------------')
            print('-----------------------------------')
            print('-----------------------------------')
            return pathFile
        else:
            pathFolder = self.get_pathFolder_ReadImage()
            return pathFolder
   
    def get_pathFile_SettingsVisualize(self):
        rootPath = Path(self.pathFolder_WriteResults)
        folderName = "Settings"
        fileName   = "Visualize.json" 
        pathFile   = Path.joinpath(rootPath, folderName, fileName)   
        return pathFile
    
    def get_pathFile_SettingsDetect(self):
        # Get Path of JSON         
        rootPath = Path(self.pathFolder_WriteResults)
        folderName = "Settings"
        fileName   = "Detect.json" 
        pathFile   = Path.joinpath(rootPath, folderName, fileName)
        return pathFile


    def get_pathFile_ScanOrigins(self):
        rootPath = Path(self.pathFolder_WriteResults)
        folderName = "Detections"
        fileName   = "0_Origins.csv"         
        pathFile   = Path.joinpath(rootPath, folderName, fileName) 
        return pathFile

    def get_pathFile_ScanDetections(self):
        rootPath = Path(self.pathFolder_WriteResults)
        folderName = "Detections"
        fileName   = "1b_MultiScale_Detections_Max.csv"         
        pathFile   = Path.joinpath(rootPath, folderName, fileName) 
        return pathFile
    
    def get_pathFile_ScanDetectionsFilter(self):
        rootPath = Path(self.pathFolder_WriteResults)
        folderName = "Detections"
        fileName   = "1c_MultiScale_Detections_Filter.csv"         
        pathFile   = Path.joinpath(rootPath, folderName, fileName) 
        return pathFile




# =============================================================================
#     Manage DataFrames (Tables)
# =============================================================================
    
    # Read a CSV Origins
    def read_ScanOrigins(self):        
        pathFile = self.get_pathFile_ScanOrigins()
        df = pd.read_csv(str(pathFile))   
        return df
    
    # Read a CSV Detections
    def read_ScanDetections(self):        
        pathFile   = self.get_pathFile_ScanDetections()         
        df = pd.read_csv (str(pathFile))   
        return df
   
    # Read a CSV Detections
    def read_ScanDetectionsFilter(self):        
        pathFile   = self.get_pathFile_ScanDetectionsFilter()          
        df = pd.read_csv (str(pathFile))   
        return df
    
# =============================================================================
#    Manage Detections     
# =============================================================================

    
    def display_Detections(self, df):  
        # Get the Detections       
        xyz = np.array(df[['X_abs_out_px', 'Y_abs_out_px', 'Z_abs_out_px']])

        # Ploting Detections as Dots
        visView = self.visView_XYZ
        visDots = self.visDots_XYZ        
        update_visDots(visView, visDots, xyz)  
        
        # =============================================================================
        #         
        # =============================================================================
        
        #Ploting Computing Cubes
        df = self.read_ScanOrigins()
        v_ScannerPosXYZ  = df[['X_ref_out_px', 'Y_ref_out_px', 'Z_ref_out_px']]
        v_ScannerSizeXYZ = df[['X_ScannerSize_Out_px', 'Y_ScannerSize_Out_px', 'Z_ScannerSize_Out_px']]
        v_ScannerPosXYZ = np.array(v_ScannerPosXYZ)
        v_ScannerSizeXYZ = np.array(v_ScannerSizeXYZ)
        
        remove_visObjects(self.visCubes)
        self.visCubes = []
        
        visView = self.visView_XYZ  
        n = v_ScannerPosXYZ.shape[0]
        for i in range(0, n): 
            
            scanner_pos  = v_ScannerPosXYZ[i,:]
            scanner_size = v_ScannerSizeXYZ[i,:]
            visCube = add_visCubeObjectToView(visView)            
            update_visCube(visView, visCube, scanner_pos, scanner_size, mode='corner')
            self.visCubes.append(visCube)
        
        self.event_pushButton_checkFilters() #???bypassing

    def get_filterExtremes(self):
        df = self.read_ScanDetections()
        
        Rmin = df['S_um'].min()
        Rmax = df['S_um'].max()
        
        Nmin = df['N'].min()
        Nmax = df['N'].max()  
        
        Imin = df['I_Raw'].min()
        Imax = df['I_Raw'].max()  
        
        Gmin = df['I_DoG'].min()
        Gmax = df['I_DoG'].max()  
        
        return Rmin, Rmax, Nmin, Nmax, Imin, Imax, Gmin, Gmax

    def apply_Filters(self):
        # Get Filter Values        
        Rmin, Rmax, Nmin, Nmax, Imin, Imax, Gmin, Gmax = self.get_filterValues()
        
        # Get Detections Table
        df = self.read_ScanDetections()
        
        # Filter
        maskR = (df['S_um']>=Rmin) & (df['S_um']<=Rmax)
        maskN = (df['N']>=Nmin) & (df['N']<=Nmax)
        maskI = (df['I_Raw']>=Imin) & (df['I_Raw']<=Imax)
        maskG = (df['I_DoG']>=Gmin) & (df['I_DoG']<=Gmax)
        mask = maskR & maskN & maskI & maskG
        df = df[mask]
        
        # Save
        pathFile = self.get_pathFile_ScanDetectionsFilter()
        df.to_csv(pathFile, sep=',', index=False, encoding='utf-8')
        
        #???Remove
        # rootPath = Path(self.pathFolder_WriteResults)
        # folderName = "Detections"
        # pathFolder_WriteResults   = Path.joinpath(rootPath, folderName)
        # fileName = '1c_MultiScale_Detections_Filter'        
        # save_CSV(df, pathFolder_WriteResults, fileName)

        # Get Detections Table
        df = self.read_ScanDetectionsFilter()
        self.display_Detections(df)
        
        
        # #Filtering
        # df = df[df['S_um']>5]
# =============================================================================
#   GUI Events that change VisPy Objects
# =============================================================================
    
    def get_visOrthoView(self):
        return self.visOrtho_XYZ, self.visOrtho_XY, self.visOrtho_YZ, self.visOrtho_XZ

    def event_ScrollBar_changeSlideXY(self, ev):
        print()
        print(ev)
        nz = ev
        
        visOrtho = self.visOrtho_XY
        img3D = self.read_resampledImage(self.pathFolder_WriteResults)
        
        #yxz (tif) to zyx (VisPy)
        img3D = np.transpose(img3D, (2,0,1))
    
        #Get Plane
        img2D = np.array([img3D[nz, :, :]])  
        
        update_visPlane(img2D, visOrtho)
        
    def event_ScrollBar_changeSlideYZ(self, ev):
        print()
        print(ev)
        nx = ev
        
        visOrtho = self.visOrtho_YZ
        img3D = self.read_resampledImage(self.pathFolder_WriteResults)
        
        #yxz (tif) to zyx (VisPy)
        img3D = np.transpose(img3D, (2,0,1))
    
        #Get Plane
        img2D = np.array([img3D[:, :, nx]])  
        
        update_visPlane(img2D, visOrtho)
        
    def event_ScrollBar_changeSlideXZ(self, ev):
        print()
        print(ev)
        ny = ev
        
        visOrtho = self.visOrtho_XZ
        img3D = self.read_resampledImage(self.pathFolder_WriteResults)
        
        #yxz (tif) to zyx (VisPy)
        img3D = np.transpose(img3D, (2,0,1))
    
        #Get Plane
        img2D = np.array([img3D[:, ny, :]])  
        
        update_visPlane(img2D, visOrtho)
    
    # def update_plane(self, ev, visView):
    #     nz = ev
    #     img3D = self.read_resampledImage(self.pathFolder_WriteResults)
    #     update_visPlane(img3D, visView, nz)
    
        
# =============================================================================
#    GUI Writes 
# =============================================================================
    # =============================================================================
    # Set - ImageView                    
    # =============================================================================
    def set_zPlaneRanges(self, xyz_min, xyz_max):
        # Unpack
        nx_min, ny_min, nz_min = xyz_min
        nx_max, ny_max, nz_max = xyz_max
        
        #Radius
        self.ui.horizontalScrollBar_XY.setMinimum(int(nz_min))
        self.ui.horizontalScrollBar_YZ.setMinimum(int(nx_min))
        self.ui.horizontalScrollBar_XZ.setMinimum(int(ny_min))
        
        self.ui.horizontalScrollBar_XY.setMaximum(int(nz_max))
        self.ui.horizontalScrollBar_YZ.setMaximum(int(nx_max))
        self.ui.horizontalScrollBar_XZ.setMaximum(int(ny_max))
        
    def set_zPlaneValues(self, xyz):
        x, y, z = xyz
        self.ui.horizontalScrollBar_XY.setValue(int(z))
        self.ui.horizontalScrollBar_YZ.setValue(int(x))
        self.ui.horizontalScrollBar_XZ.setValue(int(y))

    # =============================================================================
    #   Set - Visualize  
    # =============================================================================
    def set_checkVisIn(self, imgDimXYZ, bitDepth, fileExtension, memSize):
        
        self.ui.label_dimX_VisIn.setText('{:7.0f}'.format(imgDimXYZ[0]))
        self.ui.label_dimY_VisIn.setText('{:7.0f}'.format(imgDimXYZ[1]))
        self.ui.label_dimZ_VisIn.setText('{:7.0f}'.format(imgDimXYZ[2]))
        
        self.ui.label_bitDepth_VisIn.setText(str(bitDepth))
        self.ui.label_fileExtension_VisIn.setText(str(fileExtension))
        self.ui.label_memSize_VisIn.setText('{:0.3f}'.format(memSize))
    
    def set_ReSamplingVis(self, voxelSize_VisIn_um, voxelSize_VisOut_um):
        #Unpack
        voxelSizeX_VisIn, voxelSizeY_VisIn, voxelSizeZ_VisIn = voxelSize_VisIn_um
        voxelSizeX_VisOut, voxelSizeY_VisOut, voxelSizeZ_VisOut = voxelSize_VisOut_um
        
        #Set In
        self.ui.doubleSpinBox_voxelSizeX_VisIn.setValue(voxelSizeX_VisIn)
        self.ui.doubleSpinBox_voxelSizeY_VisIn.setValue(voxelSizeY_VisIn)
        self.ui.doubleSpinBox_voxelSizeZ_VisIn.setValue(voxelSizeZ_VisIn)

        #Set Out
        self.ui.doubleSpinBox_voxelSizeX_VisOut.setValue(voxelSizeX_VisOut)
        self.ui.doubleSpinBox_voxelSizeY_VisOut.setValue(voxelSizeY_VisOut)
        self.ui.doubleSpinBox_voxelSizeZ_VisOut.setValue(voxelSizeZ_VisOut)
    
    def set_checkVisOut(self, imgDimXYZ, bitDepth, fileExtension, memSize):

        self.ui.label_dimX_VisOut.setText('{:7.0f}'.format(imgDimXYZ[0]))
        self.ui.label_dimY_VisOut.setText('{:7.0f}'.format(imgDimXYZ[1]))
        self.ui.label_dimZ_VisOut.setText('{:7.0f}'.format(imgDimXYZ[2]))
        
        self.ui.label_bitDepth_VisOut.setText(str(bitDepth))
        self.ui.label_fileExtension_VisOut.setText(str(fileExtension))
        self.ui.label_memSize_VisOut.setText('{:0.3f}'.format(memSize))
        
    # =============================================================================
    #   Set - Select
    # =============================================================================
    def set_selectStartXYZ(self, xyz):
        #Unpack
        x, y, z = xyz        
        
        #Set StartXY
        self.ui.spinBox_selectStartX.setValue(x)
        self.ui.spinBox_selectStartY.setValue(y)
        self.ui.spinBox_selectStartZ.setValue(z)
    
    def set_selectStopXYZ(self, xyz):
        #Unpack
        x, y, z = xyz        
        
        #Set StartXY
        self.ui.spinBox_selectStopX.setValue(x)
        self.ui.spinBox_selectStopY.setValue(y)
        self.ui.spinBox_selectStopZ.setValue(z)        

    # =============================================================================
    #   Set - Detect                    
    # =============================================================================
    
    def set_ReSamplingDet(self, voxelSize_In_um, voxelSize_Out_um):
        #Unpack
        voxelSizeX_In, voxelSizeY_In, voxelSizeZ_In = voxelSize_In_um
        voxelSizeX_Out, voxelSizeY_Out, voxelSizeZ_Out = voxelSize_Out_um
        
        #Set In
        self.ui.doubleSpinBox_voxelSizeX_DetIn.setValue(voxelSizeX_In)
        self.ui.doubleSpinBox_voxelSizeY_DetIn.setValue(voxelSizeY_In)
        self.ui.doubleSpinBox_voxelSizeZ_DetIn.setValue(voxelSizeZ_In)

        #Set Out
        self.ui.doubleSpinBox_voxelSizeX_DetOut.setValue(voxelSizeX_Out)
        self.ui.doubleSpinBox_voxelSizeY_DetOut.setValue(voxelSizeY_Out)
        self.ui.doubleSpinBox_voxelSizeZ_DetOut.setValue(voxelSizeZ_Out)
    
    def set_cellRadiusRange(self, cellRadiusMin_um, cellRadiusMax_um):
        self.ui.spinBox_cellRadiusMin.setValue(int(cellRadiusMin_um))
        self.ui.spinBox_cellRadiusMax.setValue(int(cellRadiusMax_um))

    def set_nScales(self, nScales):
        self.ui.spinBox_nScales.setValue(int(nScales))
        

    def set_computingCubeSize(self, computingCubeSize):
        self.ui.spinBox_computingCubeSize.setValue(int(computingCubeSize))

    def set_computingCubeOverlap(self, computingCubeOverlap):
        self.ui.spinBox_computingCubeOverlap.setValue(int(computingCubeOverlap))

    def set_computeTensor(self, boolTensor):
        if boolTensor==True:            
            self.ui.radioButton_TensorOn.setChecked(True)
        else:
            self.ui.radioButton_TensorOff.setChecked(True)
     
    def set_nProcess(self, nProcess):
        self.ui.spinBox_nProcess.setValue(int(nProcess))
   
    # =============================================================================
    # Set - Filter                    
    # =============================================================================
    def set_filterRanges(self, Rmin, Rmax, Nmin, Nmax, Imin, Imax, Gmin, Gmax):
        #Radius
        self.ui.horizontalScrollBar_Rmin.setMinimum(int(Rmin))
        self.ui.horizontalScrollBar_Rmin.setMaximum(int(Rmax))
        self.ui.horizontalScrollBar_Rmin.setValue(int(Rmin))
        
        self.ui.horizontalScrollBar_Rmax.setMinimum(int(Rmin))
        self.ui.horizontalScrollBar_Rmax.setMaximum(int(Rmax))
        self.ui.horizontalScrollBar_Rmax.setValue(int(Rmax))
      
        #Scale
        self.ui.horizontalScrollBar_Nmin.setMinimum(int(Nmin))
        self.ui.horizontalScrollBar_Nmin.setMaximum(int(Nmax))
        self.ui.horizontalScrollBar_Nmin.setValue(int(Nmin))
        
        self.ui.horizontalScrollBar_Nmax.setMinimum(int(Nmin))
        self.ui.horizontalScrollBar_Nmax.setMaximum(int(Nmax))
        self.ui.horizontalScrollBar_Nmax.setValue(int(Nmax))
        
        #Intensity
        self.ui.horizontalScrollBar_Imin.setMinimum(int(Imin))
        self.ui.horizontalScrollBar_Imin.setMaximum(int(Imax))
        self.ui.horizontalScrollBar_Imin.setValue(int(Imin))
        
        self.ui.horizontalScrollBar_Imax.setMinimum(int(Imin))
        self.ui.horizontalScrollBar_Imax.setMaximum(int(Imax))
        self.ui.horizontalScrollBar_Imax.setValue(int(Imax))

        #Gain
        print()
        print('Gmin', Gmin)
        print('Gmax', Gmax)
        self.ui.horizontalScrollBar_Gmin.setMinimum(int(Gmin))
        self.ui.horizontalScrollBar_Gmin.setMaximum(int(Gmax))
        self.ui.horizontalScrollBar_Gmin.setValue(int(Gmin+1))
        
        self.ui.horizontalScrollBar_Gmax.setMinimum(int(Gmin))
        self.ui.horizontalScrollBar_Gmax.setMaximum(int(Gmax))
        self.ui.horizontalScrollBar_Gmax.setValue(int(Gmax))
        
        
    def set_filterValues(self, Rmin, Rmax, Nmin, Nmax, Imin, Imax, Gmin, Gmax):
        # Radius
        self.ui.horizontalScrollBar_Rmin.setValue(int(Rmin))
        self.ui.horizontalScrollBar_Rmax.setValue(int(Rmax))
        
        #N
        self.ui.horizontalScrollBar_Nmin.setValue(int(Nmin))
        self.ui.horizontalScrollBar_Nmax.setValue(int(Nmax))
        
        # Intensity
        self.ui.horizontalScrollBar_Imin.setValue(int(Imin))
        self.ui.horizontalScrollBar_Imax.setValue(int(Imax))

        # Gain
        self.ui.horizontalScrollBar_Gmin.setValue(int(Gmin))
        self.ui.horizontalScrollBar_Gmax.setValue(int(Gmax))
        
        
# =============================================================================
# 
# =============================================================================


# =============================================================================
#    GUI Gets (Reads) 
# =============================================================================

    # =============================================================================
    # Get - Visualize
    # =============================================================================

    
    def get_checkVisIn(self):
        
        dimX = int(self.ui.label_dimX_VisIn.text())
        dimY = int(self.ui.label_dimY_VisIn.text())
        dimZ = int(self.ui.label_dimZ_VisIn.text())
        dimXYZ = np.array([dimX, dimY, dimZ])
        
        bitDepth = int(self.ui.label_bitDepth_VisIn.text())
        fileExtension = self.ui.label_fileExtension_VisIn.text()
        memSize = float(self.ui.label_memSize_VisIn.text())
        
        return  dimXYZ, bitDepth, fileExtension, memSize
    
    def get_ReSamplingVis(self):
  
        #Get In
        voxelSizeX_In = self.ui.doubleSpinBox_voxelSizeX_VisIn.value()
        voxelSizeY_In = self.ui.doubleSpinBox_voxelSizeY_VisIn.value()
        voxelSizeZ_In = self.ui.doubleSpinBox_voxelSizeZ_VisIn.value()
        
        #Get Out
        voxelSizeX_Out = self.ui.doubleSpinBox_voxelSizeX_VisOut.value()
        voxelSizeY_Out = self.ui.doubleSpinBox_voxelSizeY_VisOut.value()
        voxelSizeZ_Out = self.ui.doubleSpinBox_voxelSizeZ_VisOut.value() 

        voxelSize_In_um  = np.array([voxelSizeX_In, voxelSizeY_In, voxelSizeZ_In])
        voxelSize_Out_um = np.array([voxelSizeX_Out, voxelSizeY_Out, voxelSizeZ_Out])
        return voxelSize_In_um, voxelSize_Out_um
    
    def get_checkVisOut(self):
  
        dimX = int(self.ui.label_dimX_VisOut.text())
        dimY = int(self.ui.label_dimY_VisOut.text())
        dimZ = int(self.ui.label_dimZ_VisOut.text())
        dimXYZ = np.array([dimX, dimY, dimZ])
        
        bitDepth = int(self.ui.label_bitDepth_VisOut.text())
        fileExtension = self.ui.label_fileExtension_VisOut.text()
        memSize = float(self.ui.label_memSize_VisOut.text())
        
        return dimXYZ, bitDepth, fileExtension, memSize
    
    # =============================================================================
    #  Get - Select   
    # =============================================================================
    def get_selectCropXYZ(self):
     
        #Set StartXYZ
        x = self.ui.spinBox_selectStartX.value()
        y = self.ui.spinBox_selectStartY.value()
        z = self.ui.spinBox_selectStartZ.value()
        xyzStart = np.array([x, y, z])
        
        x = self.ui.spinBox_selectStopX.value()
        y = self.ui.spinBox_selectStopY.value()
        z = self.ui.spinBox_selectStopZ.value()
        xyzStop = np.array([x, y, z])
        
        return xyzStart, xyzStop
    
    # =============================================================================
    # Get - Detect
    # =============================================================================

    def get_ReSamplingDet(self):
  
        #Get In
        voxelSizeX_In = self.ui.doubleSpinBox_voxelSizeX_DetIn.value()
        voxelSizeY_In = self.ui.doubleSpinBox_voxelSizeY_DetIn.value()
        voxelSizeZ_In = self.ui.doubleSpinBox_voxelSizeZ_DetIn.value()
        
        #Get Out
        voxelSizeX_Out = self.ui.doubleSpinBox_voxelSizeX_DetOut.value()
        voxelSizeY_Out = self.ui.doubleSpinBox_voxelSizeY_DetOut.value()
        voxelSizeZ_Out = self.ui.doubleSpinBox_voxelSizeZ_DetOut.value() 

        voxelSize_In_um  = np.array([voxelSizeX_In, voxelSizeY_In, voxelSizeZ_In])
        voxelSize_Out_um = np.array([voxelSizeX_Out, voxelSizeY_Out, voxelSizeZ_Out])
        return voxelSize_In_um, voxelSize_Out_um
    
    def get_cellRadiusRange(self):
        cellRadiusMin_um = self.ui.spinBox_cellRadiusMin.value()
        cellRadiusMax_um = self.ui.spinBox_cellRadiusMax.value()
        return cellRadiusMin_um, cellRadiusMax_um
    
    def get_nScales(self):
        nScales = self.ui.spinBox_nScales.value()
        return nScales   
    
    def get_computingCubeSize(self):
        computingCubeSize = self.ui.spinBox_computingCubeSize.value()
        return computingCubeSize
    
    def get_computingCubeOverlap(self):
        computingCubeOverlap = self.ui.spinBox_computingCubeOverlap.value()  
        return computingCubeOverlap

    def get_computeTensor(self):            
        if (self.ui.radioButton_TensorOn.isChecked()) & (not self.ui.radioButton_TensorOff.isChecked()):
            print()
            print('tensor on')
            boolTensor = True
        elif (not self.ui.radioButton_TensorOn.isChecked()) &  (self.ui.radioButton_TensorOff.isChecked()):
            print()
            print('tensor off')
            boolTensor = False
        return boolTensor
    
    def get_nProcess(self):
        nProcess = self.ui.spinBox_nProcess.value()  
        return nProcess
    
    # =============================================================================
    # Get - Filter
    # =============================================================================   
    def get_filterValues(self):
        #Radius
        Rmin = self.ui.horizontalScrollBar_Rmin.value()
        Rmax = self.ui.horizontalScrollBar_Rmax.value()
        
        #N
        Nmin = self.ui.horizontalScrollBar_Nmin.value()
        Nmax = self.ui.horizontalScrollBar_Nmax.value()
        
        #Intensity
        Imin = self.ui.horizontalScrollBar_Imin.value()
        Imax = self.ui.horizontalScrollBar_Imax.value()
        
        #Gain
        Gmin = self.ui.horizontalScrollBar_Gmin.value()
        Gmax = self.ui.horizontalScrollBar_Gmax.value()
        return Rmin, Rmax, Nmin, Nmax, Imin, Imax, Gmin, Gmax 
    

    
# =============================================================================
# Settings
# =============================================================================
    # =============================================================================
    #  Settings: Visualize
    # =============================================================================
    def save_settingsVisualize(self):  
        #Get Variables
        pathFolder_ReadImage = self.pathFolder_ReadImage
        pathFolder_WriteResults = self.pathFolder_WriteResults
        voxelSize_VisIn_um = self.voxelSize_VisIn_um
        voxelSize_VisOut_um = self.voxelSize_VisOut_um
        
        # Save as a JSON
        save_jsonVisualize(pathFolder_ReadImage, pathFolder_WriteResults, voxelSize_VisIn_um, voxelSize_VisOut_um)       
                
    def read_settingsVisualize(self): 
        # Read a JSON        
        pathFile = self.get_pathFile_SettingsVisualize()    
        pathFolder_ReadImage, pathFolder_WriteResults, voxelSize_VisIn_um, voxelSize_VisOut_um = read_jsonVisualize(pathFile) 
        return pathFolder_ReadImage, pathFolder_WriteResults, voxelSize_VisIn_um, voxelSize_VisOut_um
    
    # =============================================================================
    #  Settings: Detect
    # =============================================================================
    def save_settingsDetect(self):  
        #Get Folder
        pathFolder_ReadImage = self.get_pathFolder_ReadImage()
        pathFolder_WriteResults = self.get_pathFolder_WriteResults()
        
        #Get Selection Settings
        xyzStart, xyzStop = self.get_selectCropXYZ()
        
        #Get Image Information
        dimXYZ, bitDepth, fileExtension, memSize = self.get_checkVisIn()
        
        #Get Detection Settings
        voxelSize_DetIn_um, voxelSize_DetOut_um = self.get_ReSamplingDet()
        cellRadiusMin_um, cellRadiusMax_um = self.get_cellRadiusRange()
        nScales = self.get_nScales()
        computingCubeSize = self.get_computingCubeSize()
        computingCubeOverlap = self.get_computingCubeOverlap()
        computeTensor = self.get_computeTensor()
        nProcess = self.get_nProcess()
        
        
        # Save as a JSON
        save_jsonDetect(pathFolder_ReadImage, pathFolder_WriteResults,
                        xyzStart, xyzStop, dimXYZ,
                        voxelSize_DetIn_um, voxelSize_DetOut_um,
                        cellRadiusMin_um, cellRadiusMax_um, nScales,
                        computingCubeSize, computingCubeOverlap,
                        computeTensor,
                        nProcess
                        ) 
        

    def read_settingsDetect(self, ): 
        # Read a JSON        
        pathFile = self.get_pathFile_SettingsDetect()    
        args = read_jsonDetect(pathFile) 
        return args
  
# =============================================================================
# Image Manager
# =============================================================================
    def save_resampledImage(self, img3D, pathFolder_WriteResults):
        print()
        print('Start: save_resampledImage()')
  
        #Save Resampled Image as a 3D tiff
        pathFolder = self.get_pathFolder_ImageResampled()
        # createFolder(str(pathFolder), remove=False) # !!! Fails
        createFolder(str(pathFolder), remove=True) #!!!
        fileName   = "Visualize"
        # print()
        # print('--------------------------')
        # print('img3D', img3D)
        # print('img3D', img3D)
        # print('str(pathFolder)', str(pathFolder))
        save3Dimage_as3DStack(img3D, pathFolder, fileName)
        
        print()
        print('Stop: save_resampledImage()')
        
    def read_resampledImage(self, pathFolder_WriteResults):
        #Read Resampled Image as a 3D tiff
        # pathFile = self.get_pathFile_ImageResampled()
        pathFile = self.get_pathFile_ImageResampled(self.isResampled)
        
        
        print()
        print('read_resampledImage()')
        print('pathFile: \n', pathFile)
        # Bypass
        # pathFile = str(Path(r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\ImageTileMini'))
        
        img3D = read_Image(pathFile, nThreads=1)
        return img3D
        

    def set_img3D(self, img3D):
        self.img3D = img3D
        
    def get_img3D(self):
        img3D = self.img3D
        return img3D        

# =============================================================================
#   Computations        
# =============================================================================

    # =============================================================================
    #   Computation: ReSampling  
    # =============================================================================
    def computeResampledImage(self):
        print()
        print('Start: computeResampledImage()')

        # Read JSON
        pathFolder_ReadImage, pathFolder_WriteResults, voxelSize_In_um, voxelSize_Out_um = self.read_settingsVisualize()
        
        # print()
        # print('JSON Visualize Settings')
        # print('pathFolder_ReadImage:', pathFolder_ReadImage)
        # print('pathFolder_WriteResults:', pathFolder_WriteResults)
        # print('voxelSize_In_um:', voxelSize_In_um)
        # print('voxelSize_Out_um:', voxelSize_Out_um)
        
        # Read ImgIn
        img3D = read_Image(pathFolder_ReadImage, nThreads=1)

        # Get the Out/In Ratio 
        r_um = voxelSize_Out_um/voxelSize_In_um
        r_px = 1/r_um
        
        # Compute the Output Dimensions
        imgDimYXZ_In = np.array(img3D.shape)  
        imgDimXYZ_In = imgDimYXZ_In[[1,0,2]]
        imgDimXYZ_Out = np.round(imgDimXYZ_In*r_px).astype(int)
        
        print()
        print('Size')
        print('imgDimXYZ_In: ', imgDimXYZ_In) 
        print('imgDimXYZ_Out:', imgDimXYZ_Out) 
        print('r_um:', r_um)
        # print('r_px:', r_px)
        
        #Test
        # a = np.array([4.44444444, 4.44444444, 1.        ])
        # a = np.array([1., 1., 1.        ])
        # (a[0]!=1)&(a[1]!=1)&(a[2]!=1)
        # (a[0]!=1)|(a[1]!=1)|(a[2]!=1)
        
        # Resample Image
        if (r_um[0]!=1)|(r_um[1]!=1)|(r_um[2]!=1): 
            print()
            print('Compute Resampling...')
            [img3D, start, stop]  = resample_3DImage(img3D, imgDimXYZ_Out)
    
        
        # Save Image        
        self.save_resampledImage(img3D, pathFolder_WriteResults)
        
        
        print()
        print('Stop: computeResampledImage()')
        # jajajaj
        
        return pathFolder_WriteResults

    def compute_cropImage(self, img3D_bk, xyzStart, xyzStop):
        img3D = img3D_bk.copy()
        print()
        print('img3D.min()', img3D.min())
        print('img3D.max()', img3D.max())
        
        x0, y0, z0 = xyzStart
        x1, y1, z1 = xyzStop
        
        #Op1
        # img3D = img3D[y0:y1+1, x0:x1+1, z0:z1+1 ]
        # img3D = img3D[x0:x1+1, y0:y1+1, z0:z1+1]
        # img3D = img3D[z0:z1+1, y0:y1+1, x0:x1+1]
        
        #??? Op2 ???
        mask = np.ones((img3D.shape), dtype=bool)
        x = np.arange(x0, x1+1)
        y = np.arange(y0, y1+1)
        z = np.arange(z0, z1+1)
        ux, uy, uz = np.meshgrid(x,y,z)
        mask[uy, ux, uz] = False
        I_min = img3D[np.invert(mask)].min()
        img3D[mask] = I_min
        
        print()
        print('img3D.min()', img3D.min())
        print('img3D.max()', img3D.max())
        
        return img3D


    # =============================================================================
    #   Computation: Detection 
    # =============================================================================
    def compute_scanDetection(self):
        print()
        print('Start: compute_Detection()')
                
        # Read JSON
        pathFile = self.get_pathFile_SettingsDetect()
        
        #Start Detections
        run_scanDetections(pathFile)
        
        
    
# M = np.array([[1,2,3], [4,5,6], [7,8,9]])       
# =============================================================================
#   Visualizations
# =============================================================================
    def get_visViewBox(self):        
        visViewBox_XYZ = self.visView_XYZ 
        visViewBox_XY = self.visView_XY
        visViewBox_YZ = self.visView_YZ
        visViewBox_XZ = self.visView_XZ
        
        return visViewBox_XYZ, visViewBox_XY, visViewBox_YZ, visViewBox_XZ
    
   
               
# =============================================================================
#  Long Task Computations   
# =============================================================================


    # def workEvent_ComputeResample_readResults(self, res):
    #     print()
    #     print('event_readResults')
    #     print(res)
        
    def workEvent_ComputeResample_finished(self):
        print()
        print('Start: workEvent_ComputeResample_finished()')
        
        # PushButton: Compute ReSampling
        self.event_pushButton_displayResampledImage() #????
        
        print()
        print('Stop: workEvent_ComputeResample_finished()')
        
        
    def work_onComputeResampledImage(self):
 
        # Step 1: Assign function and args
        # computeResampledImage
        fn = self.computeResampledImage
        # fn = self.sumNumber
        # args = []
        
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        
        # Step 3: Create a worker object
        self.worker = Worker(fn)
        
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # Step 6: Connect singals 
        # self.worker.progress.connect(self.event_workerPseudoStitched_reaProgress)
        # self.worker.results.connect(self.workEvent_ComputeResample_readResults)
        self.worker.finished.connect(self.workEvent_ComputeResample_finished)
        
        # Step 7: Start the thread
        self.thread.start()
        
        #See after processing
        
    def workEvent_ComputeScanDetections_finished(self):
        print()
        print('Start: workEvent_ComputeScanDetections_finished()')
        
        self.ui.pushButton_computeDetections.setEnabled(True)
        self.ui.pushButton_displayDetections.setEnabled(True)
                
        print()
        print('Stop: workEvent_ComputeScanDetections_finished()') 
        self.event_pushButton_displayDetections()
        
    def work_onComputeScanDetections(self):
 
        # Step 1: Assign function and args
        fn = self.compute_scanDetection
        # fn = self.sumNumber
        # args = []
        
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        
        # Step 3: Create a worker object
        self.worker = Worker(fn)
        
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # Step 6: Connect singals 
        # self.worker.progress.connect(self.event_workerPseudoStitched_reaProgress)
        # self.worker.results.connect(self.workEvent_ComputeScanDetections_readResults)
        self.worker.finished.connect(self.workEvent_ComputeScanDetections_finished)
        
        # Step 7: Start the thread
        self.thread.start()
        
        #See after processing         
         
         

# =============================================================================
#         
# =============================================================================
# class PandasModel(QtCore.QAbstractTableModel):
#     def __init__(self, data, parent=None):
#         QtCore.QAbstractTableModel.__init__(self, parent)
#         self._data = data

#     def rowCount(self, parent=None):
#         return len(self._data.values)

#     def columnCount(self, parent=None):
#         return self._data.columns.size

#     def data(self, index, role=QtCore.Qt.DisplayRole):
#         if index.isValid():
#             if role == QtCore.Qt.DisplayRole:
#                 return QtCore.QVariant(str(self._data.values[index.row()][index.column()]))
#         return QtCore.QVariant()        



    
    # def sort(self, column, order):
    #     if order == 0:
    #         self._dataframe = self._dataframe.reindex(index=order_by_index(self._dataframe.index, index_natsorted(self._dataframe[column])))
    #     else:
    #         self._dataframe = self._dataframe.reindex(index=order_by_index(self._dataframe.index, reversed(index_natsorted(self._dataframe[column]))))
    
    #     self._dataframe.reset_index(inplace=True, drop=True)
    #     self.setDataFrame(self._dataframe)        
#==============================================================================
#            MAIN
#==============================================================================
if __name__ == '__main__':
    #Prevents issues with pyinstaller
    freeze_support()
    
    # checks if QApplication already exists 
    app = QtWidgets.QApplication.instance() 
    if not app: 
        # create QApplication if it doesnt exist 
         app = QtWidgets.QApplication([])   
    # to cause the QApplication to be deleted later
    app.aboutToQuit.connect(app.deleteLater)    
    
    application = mywindow()     
    application.show()     
    sys.exit(app.exec_())

# =============================================================================
# 
# =============================================================================

# class ThreadClass(QtCore.QThread):
#     def __init__(self, parent=None):
#         super(ThreadClass, self).__init__(parent)
        
#     def run(self):
#         for i in range(0, 10):
#             pass    
# =============================================================================
# Draft
# =============================================================================
        
        
    # def on_mouse_press0(self, ev):
    #     print()
    #     print('on_mouse_press0')
    #     visView, visCam, visVol, visBox, visAxes  = self.visView_XYZ, self.visCam_XYZ, self.visVol_XYZ, self.visBox_XYZ, self.visAxes_XYZ 
    #     biestableClick = self.biestableClick
        
    #     if biestableClick==True:
    #         biestableClick = False
    #         #Read Resampled Image as a 3D tiff
    #         pathFile    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\ImageWholeBrainx32'
    #         # pathFile    = r'C:\Users\aarias\MySpyderProjects\p6_Cell_v11\Results\mainTest\ImageResampled\Visualize.tif'
    #         pathFile   = Path(pathFile)
    #         img3D = read_Image(pathFile, nThreads=1)
    #         # update_visVol(visView, visVol, img3D)
    #         # update_vis3DDisplay(visView, visCam, visVol, visBox, img3D)
    #         update_visDisplay(visView, visCam, visVol, visBox, visAxes, img3D)
                        
    #     else:                
    #         biestableClick = True 
    #         #Read Resampled Image as a 3D tiff
    #         pathFile    = r'C:\Users\aarias\MyPipeLine\MiniTest\ScansToPose\Image\ImageTileMini'
    #         pathFile   = Path(pathFile)
    #         img3D = read_Image(pathFile, nThreads=1)
    #         # update_visVol(visView, visVol, img3D)
    #         # update_vis3DDisplay(visView, visCam, visVol, visBox, img3D)

    #         update_visDisplay(visView, visCam, visVol, visBox, visAxes, img3D)
        
    #     self.biestableClick = biestableClick 

    # def on_mouse_press(self, ev):
    #     print()
    #     print('on_mouse_press')        
    #     tab_ix = self.ui.tabWidget.currentIndex()        
    #     print('Current Tab:', tab_ix)
    #     print('ev.button:', ev.button)
        
        
    #     if (ev.button==1) & (tab_ix==1):
    #         view = self.visView_XY  
    #         biestableClick = self.biestableClick
    #         visSelectPoint = self.visSelectPoint
    #         visSelectBox = self.visSelectBox
            
    #         # Get the Out/In Ratio 
    #         voxelSize_VisIn_um, voxelSize_VisOut_um = self.get_ReSamplingVis()
    #         r_um = voxelSize_VisOut_um/voxelSize_VisIn_um
    #         r_px = 1/r_um
            
    #         #Transform to the World Coordinates
    #         tf = view.scene.transform
    #         xyz_Resampled = tf.imap(ev.pos)
    #         xyz_Resampled = np.round(xyz_Resampled[0:3]).astype(np.uint32)
            
    #         #Transform relative to Resampled Image
    #         xyz_Original = r_um*xyz_Resampled
    #         xyz_Original = (np.round(xyz_Original)).astype(np.uint32) # ??? +2
    #         print()
    #         print('xyz_Resampled:', xyz_Resampled)  
    #         print('xyz_Original:', xyz_Original)  
            
    #         # dimXYZ, bitDepth, fileExtension, memSize = self.get_checkVisOut()
    #         # dimZYX = dimXYZ[[2, 1, 0]]
    #         # update_visBox(visView=view, visBox=visSelectCropRect, dimZYX=dimZYX)
            
    #         # v_p = np.array([[100, 100, 0]])
    #         # v_p = r_px*v_p
    #         # v_p = np.array([xyz_Resampled])
    #         # visDots = plot_visDots(view, v_p, R=10)
                  
    #         # xy = np.array([xyz_Resampled[0], xyz_Resampled[1]])
    #         # xy_norm = np.array([xyz_Original[0], xyz_Original[1]]) 
    #         if biestableClick==True:
    #             biestableClick = False
                
    #             #Add Rectangle
    #             view.add(visSelectPoint)
    #             visSelectPoint.set_data(np.array([xyz_Resampled]),
    #                                         face_color = [0, 1, 0, 0.5],
    #                                         size=5
    #                                          )                
    #             self.set_selectStartXYZ(xyz_Original)
                
    #         else:                
    #             biestableClick = True
                
    #             #Remove Point
    #             view.add(visSelectPoint)
    #             visSelectPoint.parent = None                
    #             self.set_selectStopXYZ(xyz_Original)
                
                
    #             xyzStart, xyzStop = self.get_selectCropXYZ()
    #             xyzStart = (np.round(r_px*xyzStart)).astype(np.uint32)
    #             xyzStop = (np.round(r_px*xyzStop)).astype(np.uint32)
    #             dimXYZ = xyzStop - xyzStart
                
                
    #             view.add(visSelectBox)
    #             tx = xyzStart[0] + dimXYZ[0]//2
    #             ty = xyzStart[1] + dimXYZ[1]//2
    #             tz = xyzStart[2] + dimXYZ[2]//2
    #             t = np.array([tx, ty, tz])
                
    #             sx = dimXYZ[0]
    #             sy = dimXYZ[1]
    #             sz = dimXYZ[2]
    #             s = np.array([sx,sy, sz])
    #             print()
    #             print('hello')
    #             print(t)
    #             print(s)
    #             visSelectBox.transform = STTransform(translate=(tx, ty, tz),
    #                                                                  scale=(sx, sy, sz))
    #             visSelectBox.order = 2
    #             # visSelectCropRect(width=20, height=20, depth=20)
                
    #             # self.add_visRectangle(view, visSelectCropRect, xyStart, xyStop)

    #             # visSelectCropRect.transform.scale = sc
              
    #         self.biestableClick = biestableClick    
           
    #         print()
    #         print('biestableClick:', self.biestableClick)        
        