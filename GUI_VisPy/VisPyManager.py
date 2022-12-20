# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:15:59 2021

@author: aarias
"""
#QT GUI 
from PyQt5 import QtWidgets 

#GPU Scientific Visualization 
import vispy
from vispy import scene
from vispy.visuals.transforms import STTransform, MatrixTransform, ChainTransform
import vispy.io as io
from vispy.visuals import transforms

import numpy as np



# Forcing the Backend PyQt5
vispy.use('pyqt5')


# =============================================================================
# 
# =============================================================================
def remove_visObjects(visObjects):
    print()
    print('visObjects', visObjects)
    if visObjects:
        print()
        print('-------------------------')
        for visObject in visObjects:
            visObject.parent = None
            
            
# =============================================================================
# Adapt VisPy to QT
# =============================================================================
def embed_visViewBoxIntoQtWidget(visWidget):
    # Embed the VisPy Canvas into the GUI Widget 
    visWidget.setLayout(QtWidgets.QVBoxLayout())        
    visCanvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor = 'k')
    visWidget.layout().addWidget(visCanvas.native)
          
    # Set up a viewbox to display the image with interactive pan/zoom
    visViewBox = visCanvas.central_widget.add_view()
    
    # return visViewBox
    return visViewBox, visCanvas


# =============================================================================
# Add Cam to ViewBox
# =============================================================================
# we assume that the data is in zyx order since image data often is.
def add_VisCamToViewBox(visViewBox, fov=60.0, camType='Turntable'):
    
    if camType=='Fly':            
        cam = scene.cameras.FlyCamera(parent=visViewBox.scene, fov=fov, name='Fly')
    elif camType=='Turntable': 
        cam = scene.cameras.TurntableCamera(parent=visViewBox.scene, fov=fov, name='Turntable')
    elif camType=='Arcball':
        cam = scene.cameras.ArcballCamera(parent=visViewBox.scene, fov=fov, name='Arcball')
    # Set 2D camera (the camera will scale to the contents in the scene)
    elif camType=='PanZoom':        
        cam = scene.PanZoomCamera(parent=visViewBox.scene, aspect=1)

    else:
        print()
        print('set_visCam(), unknow camera type')
      
    # Insert Cam in View
    visViewBox.camera = cam
    
    return visViewBox.camera

def update_VisCam(visView, imgDimZYX, originXYZ=[0,0,0], mode='Start'):
    [sz, sy, sx] = imgDimZYX
    [x, y, z] = originXYZ
    print()
    print('update_VisCam()')
    print('originXYZ',originXYZ)
    print('imgDimZYX', imgDimZYX)
    # print('visView.camera.get_state()', visView.camera.get_state())
    # print('visView.camera.zoom()', visView.camera.zoom)
    # state_0 = visView.camera.get_state()
    if mode=='Start':
        if sz>1:   
            print('VisCam: 3D')
            visView.camera.set_range(x=[x, sx], y=[y, sy], z=[z, sz], margin=0.5)
            visView.camera.center = (sx//2 + x, sy//2 + y, sz//2 + z)
        else:
            print('VisCam: 2D')
            visView.camera.set_range(x=[x, sx], y=[y, sy], margin=0.5)
            visView.camera.center = (sx//2 + x, sy//2 + y)
    elif mode=='Explore':
        pass
    elif mode=='Pan':
        pass        
    else:
        print()
        print('Module: GUI_VisPy: VisPyManager ')
        print('Function: update_Viscam')
        print('mode not known')
        
    # visView.camera.set_state(state_0)
     
# =============================================================================
#    Plot Volume  
# =============================================================================

def add_visVolObjectToView(visView):
    print()
    print('START: add_visVolObjectToView()')
    visVol = scene.visuals.Volume(np.ones((100, 100, 100)), 
                                  # method='attenuated_mip', ##update vispy
                                  parent=visView)                                          
    
    print()
    print('STOP: add_visVolObjectToView()')
    return visVol

def update_visVol(visView, visVol, img3D, clim=None): 
    # img3D[img3D>1000]=0
    if clim==None:
        clim = (img3D.min(), img3D.max()) #?????
        print()
        print('clim:', clim)
        # clim = (1.0*img3D.mean(), img3D.mean() + 5*img3D.std())
        # clim = (100, 100 + 9*img3D.std())
        #clim = (0.0*img3D.mean(), img3D.mean() + 3*img3D.std())
    
    visVol.parent = None            
    visVol.set_data(img3D, clim=clim) 
    visView.add(visVol)
    # imgDimZYX = img3D.shape
    # update_VisCam(visView, imgDimZYX)



# =============================================================================
#     Plot Box 
# =============================================================================
def add_visBoxObjectToView(visView):
    # x, y, z = np.zeros(3)
    [sz, sy, sx] = [1, 1, 1]
    visBox = scene.visuals.Box(width=sx, height=sz, depth=sy,
                                color=None, #(1, 1, 1, 0)
                                # edge_color=(0, 1, 0, 1),
                                edge_color=(1, 1, 1, 0.5),
                                parent=visView.scene
                                ) 
                                
    # visBox.transform = STTransform(translate=(sx/2 + x , sy/2 + y, sz/2 + z), scale=(sx, sy, sz))
    return visBox

def update_visBox(visView, visBox, dimZYX, xyz=np.array([0,0,0])):
    x, y, z = xyz
    [sz, sy, sx] = dimZYX 
    visBox.transform = STTransform(translate=(sx/2 + x , sy/2 + y, sz/2 + z), scale=(sx, sy, sz))
    visBox.set_gl_state('translucent', depth_test=False)  

    # visView.add(visBox) #??? Important to add
     
# =============================================================================
# 
# =============================================================================
def add_visCubeObjectToView(visView):
    eColor = (1, 1, 1, 0.1)
    eColor = (1, 1, 1, 0.5)  #Grey
    eColor = (1, 0, 1, 0.15) #magenta
    eColor = (1, 1, 0, 0.15) #yellow
    visCube = scene.visuals.Cube(color=None, edge_color=eColor)
    return visCube

def update_visCube(visView, visCube, posXYZ, dimXYZ, mode='center'):
    visCube.parent = None    
    sx, sy, sz = dimXYZ
    x, y , z = posXYZ
    if mode=='center':
        visCube.transform = STTransform(translate=(x, y, z), scale=(sx, sy, sz))
    elif mode=='corner':
        visCube.transform = STTransform(translate=(sx/2 + x , sy/2 + y, sz/2 + z), scale=(sx, sy, sz))
    else:
        print()
        print('Module: GUI_VisPY\VisPyManager')
        print('Function: add_visCubeObjectToView')
        print('Warning: mode not known')
    visCube.set_gl_state('translucent', depth_test=False) 
    visView.add(visCube)
 
# =============================================================================
#     
# =============================================================================


# =============================================================================
# 
# =============================================================================
def add_visGridObjectToView(visView):
    xv, yv, zv = np.meshgrid([2,3], [2,3], [2,3])
    # visGridMesh = scene.visuals.GridMesh(xv, yv, zv)   
    # visGridMesh = scene.visuals.GridLines() 
    visGridMesh = scene.visuals.Cube(color=None, edge_color=(1, 1, 1, 0.5))
    return visGridMesh

def update_visGrid(visView, visGridMesh, posXYZ, dimXYZ):
    visGridMesh.parent = None
    
    sx, sy, sz = dimXYZ
    x, y , z = posXYZ
    visGridMesh.transform = STTransform(translate=(sx/2 + x , sy/2 + y, sz/2 + z), scale=(sx, sy, sz))
    visGridMesh.set_gl_state('translucent', depth_test=False) 
    visView.add(visGridMesh)
    
def remove_visGrid(visView, visGridMesh):
    visGridMesh.parent = None    

# =============================================================================
# # Plot XYZAxis
# =============================================================================
# Axes are x=red, y=green, z=blue.
def add_visAxesObjectToView(visView):
    visXYZAxes = scene.visuals.XYZAxis(width=2, 
                                        parent=visView.scene #??? remove this
                                       )

    return visXYZAxes

def update_visAxes(visView, visAxes, traslate=np.zeros(3), scale=np.ones(3), rotate=np.identity(3), isText=True, dpi=10):

    # Create the Affine Matrix : Traslation, Rotation, Scaling
    [x, y, z] = traslate
    M = np.concatenate((rotate.T, np.asarray([[x, y, z]])), axis=0)
    M = np.concatenate((M, np.asarray([[0, 0, 0, 1]]).T), axis=1)
    M[0] = scale[0]*M[0]
    M[1] = scale[1]*M[1]
    M[2] = scale[2]*M[2]    
    
    # print()
    # print('M:\n', M)
    # trans = visView.scene.transform
    # print('visView.scene.transform.matrix', trans.matrix)
    
    mt = MatrixTransform(matrix=M)
    visAxes.transform = mt
    # visView.add(visXYZAxes) #??? Important to add

    if scale[2]>1:        
        vol = np.prod(scale)
        l = (np.ceil(vol**(1/3))).astype(int)
        dpi = 5*l*dpi
        
    isText = False
    if isText==True:
        xyz = scale.copy()
        
        # pad = 0.25*(xyz[np.nonzero(xyz)]).min()
        # pad = 0.25*(xyz.mean())
        # pad = np.round(0.25*(np.sort(xyz)[1:])).astype(int)
        # pad = 0.25*(np.sort(xyz)[1:]).mean()
        # pad = 0.25*(xyz.min())
        pad = 0.15*(xyz[xyz>1]).min()
        
        # print('')
        # print('xyz \n', xyz)
        # print('pad \n', pad)
        
        zyxZ = [xyz[0] + pad, 0, 0]
        label = 'X'
        plot_Text(visView, zyxZ, label, c='r', dpi=dpi)
        
        zyxY = [0, xyz[1] + pad, 0]
        label = 'Y'
        plot_Text(visView, zyxY, label, c='g', dpi=dpi)    
        
        zyxX = [0, 0, xyz[2] + pad ]
        label = 'Z'
        plot_Text(visView, zyxX, label, c='b', dpi=dpi) 

    # print()
    # print('Stop: plot_visXYZAxis()')
    return visAxes
# =============================================================================
# 
# =============================================================================
def add_visTextObjectToView(visView):
    visText = scene.visuals.Text(visView.parent)    
    return visText

def update_visText(visView, visText, xyz, v_label, c='r', dpi=100):
    visText.parent = None
    visText.set_data(pos=xyz, 
                     text=v_label,
                     color=c,
                     font_size=dpi,) 
    visView.add(visText)

# Plot Text
def plot_Text(visViewBox, xyz, label, c='w', dpi=100):
    visText= scene.visuals.Text(pos=xyz, 
                                text=label,
                                color=c,
                                font_size=dpi,
                                parent=visViewBox.scene) 
    return visText


# =============================================================================
# 
# =============================================================================

def add_visDotsObjectToView():
    visDots = scene.visuals.Markers()    
    return visDots

def update_visDots(visView, visDots, v_p, R=4, dpi=90, isText=True, c=[0, 1, 0, 0.25]):
    visDots.parent = None
    if len(v_p.shape)==1:
        v_p = np.array([v_p])
    visDots.set_data(pos=v_p, 
                     size=R, 
                     face_color=c
                    ) 
    # visDots.transform = STTransform(translate=(0, 0, -200)) 
    visView.add(visDots)
    

#Plot Dots (x,z,y)
def plot_visDots(visView, v_p, R, dpi=90, isText=True, c=[0, 1, 0, 1.0]):        
    pos = v_p
    pos = pos.astype(np.int32)
    visDots = scene.visuals.Markers(pos=pos, 
                                    size=5, 
                                    # face_color=[0, 1, 0, 0.5],
                                    face_color=c,
                                    parent=visView.scene) 
    
    # Plot Text  
    if isText==True:
        ix = np.arange(0, v_p.shape[0])
        ixText = ix.astype(str)
        visText= scene.visuals.Text(pos=pos, 
                                    text=ixText ,
                                    font_size=15*dpi,
                                    parent=visView.scene)  
    return visDots

# =============================================================================
# 
# =============================================================================
def add_visLineObjectToView():
    visLine = scene.visuals.Line()  
    return visLine

def update_visLine(visView, visLine, vertex, c=[0, 1, 0, 0.25]):
    visLine.parent = None
    # vertex = np.array([0,0, 100, 100])
    visLine.set_data(pos=vertex, 
                     color=c,
                     width=4,
                    ) 
    visLine.transform = STTransform(translate=(0, 0, -1))
    visView.add(visLine)

# =============================================================================
# 
# =============================================================================


#Plot Vectors
def plot_visVector(visViewBox, p1, v, k=1.0): 
    # print()
    # print("plot_visVector")

    if len(v.shape)==1:
        m = np.sqrt((v**2).sum())
        v = k*v/m 
        p2 = p1 + v 
        arrows   = np.array([np.concatenate((p1, p2))]).astype('float')           
        posPairs = np.array(arrows.reshape((2, 3))).astype('float')
    else:            
        m = np.sqrt((v**2).sum(axis=1))            
        v = k*(v.T/m).T
        p2 = p1 + v    
        arrows   = np.column_stack((p1, p2))
        posPairs = arrows.reshape((2*arrows.shape[0], 3))     
    visArrows = scene.visuals.Arrow(pos=posPairs,                                   
                                    color=[0, 1, 1, 0.5], #red with transparency
                                    connect='segments',
                                    width=4,
                                    
                                    
                                    arrows=arrows,
                                    # arrow_type='triangle_30',
                                    arrow_type="stealth",
                                    arrow_color=[0, 0, 1, 0.5], #red with transparency
                                    arrow_size=1,
                                    
                                    method='gl',
                                    parent=visViewBox.scene)
    visArrows.transform = STTransform(translate=(0,0,0), scale=(1, 1, 1))
    visArrows.set_gl_state('translucent', depth_test=False)




 

# =============================================================================
#     
# =============================================================================
def plot_visSphere(self, view, p, r, c='g', verbose=False):
    if verbose== True:
        print()
        print("plot_visSphere")
        print('yxz=', p)
        print('radius=', r)
    
    
    if len(c)==1:
        if c=='g':
            edge_color = [0, 1, 0, 0.025]
        elif c=='r':
            edge_color = [1, 0, 0, 0.025]
        elif c=='b':
            edge_color = [0, 0, 1, 0.025] 
    else:
        edge_color = [c[0], c[1], c[2], 0.025]
        
        
        
    visSphere = scene.visuals.Sphere(radius=r, 
                                     method='latitude',
                                     color=None,                                           
                                     edge_color=edge_color,
                                     parent=view.scene,)
    visSphere.transform = STTransform(translate=(p), scale=(1, 1, 1))
    visSphere.set_gl_state('translucent', depth_test=False)

# Plot Text
def plot_Text(visViewBox, xyz, label, c='w', dpi=100):
    visText= scene.visuals.Text(pos=xyz, 
                                text=label,
                                color=c,
                                font_size=dpi,
                                parent=visViewBox.scene) 
    return visText


# =============================================================================
# 
# =============================================================================

def add_visDisplay(visWidgets, camType='Turntable'):
    visView, _ = embed_visViewBoxIntoQtWidget(visWidgets)
    visCam = add_VisCamToViewBox(visView, fov=60.0, camType=camType)
    visVol= add_visVolObjectToView(visView)
    visBox = add_visBoxObjectToView(visView) 
    visAxes = add_visAxesObjectToView(visView)
    return visView, visCam, visVol, visBox, visAxes

def update_visDisplay(visView, visCam, visVol, visBox, visAxes, img3D, mode='Start'):  
    
    imgDimZYX = np.array(img3D.shape)
    imgDimXYZ = imgDimZYX[[2, 1, 0]]
    
    update_visVol(visView, visVol, img3D)
    update_visBox(visView, visBox, imgDimZYX)
    update_VisCam(visView, imgDimZYX, mode=mode)
    # update_visAxes(visView, visAxes, scale=imgDimXYZ)
    
 
# =============================================================================
# 
# =============================================================================
    
def add_visOrthogonal(visWidgets):
    #Unpacking
    visWidget_XYZ, visWidget_XY, visWidget_YZ, visWidget_XZ = visWidgets
    
    #Embeded
    visView_XYZ, _ = embed_visViewBoxIntoQtWidget(visWidget_XYZ)
    visView_XY,  _  = embed_visViewBoxIntoQtWidget(visWidget_XY)
    visView_YZ,  _  = embed_visViewBoxIntoQtWidget(visWidget_YZ)
    visView_XZ,  _  = embed_visViewBoxIntoQtWidget(visWidget_XZ)
     
    visView = visView_XYZ
    visCam_XYZ = add_VisCamToViewBox(visView, fov=60.0, camType='Turntable')
    visVol_XYZ = add_visVolObjectToView(visView)
    visBox_XYZ = add_visBoxObjectToView(visView)

    visView = visView_XY
    visCam_XY = add_VisCamToViewBox(visView, fov=60.0, camType='PanZoom')
    visVol_XY = add_visVolObjectToView(visView)
    visBox_XY = add_visBoxObjectToView(visView)

    visView = visView_YZ
    visCam_YZ = add_VisCamToViewBox(visView, fov=60.0, camType='PanZoom')
    visVol_YZ = add_visVolObjectToView(visView)
    visBox_YZ = add_visBoxObjectToView(visView)

    visView = visView_XZ
    visCam_XZ = add_VisCamToViewBox(visView, fov=60.0, camType='PanZoom')
    visVol_XZ = add_visVolObjectToView(visView)
    visBox_XZ = add_visBoxObjectToView(visView)

    visOrtho_XYZ = visView_XYZ, visCam_XYZ, visVol_XYZ, visBox_XYZ
    visOrtho_XY  = visView_XY,  visCam_XY,  visVol_XY,  visBox_XY
    visOrtho_YZ  = visView_YZ,  visCam_YZ,  visVol_YZ,  visBox_YZ
    visOrtho_XZ  = visView_XZ,  visCam_XZ,  visVol_XZ,  visBox_XZ
    
    return visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ

def update_visOrthoView(img3D, visOrtho_XYZ, visOrtho_XY, visOrtho_YZ, visOrtho_XZ, mode='Start'):
    
    print()
    print('START: update_visOrthoView()')
    
    #yxz (tif) to zyx (VisPy)
    img3D = np.transpose(img3D, (2,0,1))

    #Get Orthogonal
    # [nz, ny, nx] = img3D.shape  
    imgDimZYX = np.array(img3D.shape)
    [nz, ny, nx] = imgDimZYX
    imgDimXYZ = imgDimZYX[[2, 1, 0]] 
      
    img2D_XY = np.array([img3D[nz//2, :,       :   ]])
    img2D_YZ = np.array([img3D[:,     :,     nx//2]])
    img2D_XZ = np.array([img3D[:,     ny//2,   :   ]])
        
    
    # VisPlot: 3D XYZ  
    visView, visCam, visVol, visBox, visAxes = visOrtho_XYZ 
    update_visDisplay(visView, visCam, visVol, visBox, visAxes, img3D, mode=mode)
    update_visAxes(visView, visAxes, scale=imgDimXYZ)
 
    # VisPlot: 2D XY  
    visView, visCam, visVol, visBox, visAxes = visOrtho_XY 
    update_visDisplay(visView, visCam, visVol, visBox, visAxes, img2D_XY, mode=mode)
    update_visAxes(visView, visAxes, scale=imgDimXYZ)
    
    # VisPlot: 2D YZ  
    visView, visCam, visVol, visBox, visAxes = visOrtho_YZ 
    update_visDisplay(visView, visCam, visVol, visBox, visAxes, img2D_YZ, mode=mode)
    Mr_X  = np.matrix([[0,   0,  -1],
                        [0,   1,   0],
                        [+1,  0,   0]])

    Mr_Y  = np.matrix([[1,  0,  0],
                        [0,  0, +1],
                        [0, -1,  0]])
    Mr = Mr_X*Mr_Y
    update_visAxes(visView, visAxes, scale=imgDimXYZ, rotate=Mr)
    
    # VisPlot: 2D XZ  
    visView, visCam, visVol, visBox, visAxes = visOrtho_XZ 
    update_visDisplay(visView, visCam, visVol, visBox, visAxes, img2D_XZ, mode=mode)
    Mr_Y     = np.matrix([[1,  0,  0],
                          [0,  0, +1],
                          [0, -1,  0]])
    Mr = Mr_Y
    # Mr = np.transpose(Mr, (2,0,1))
    update_visAxes(visView, visAxes, scale=imgDimXYZ, rotate=Mr)
    
    print()
    print('STOP: update_visOrthoView()')


def update_visPlane(img2D, visOrtho, mode='Explore'):    
    # VisPlot: 2D XY
    visView, visCam, visVol, visBox, visAxes = visOrtho
    update_visDisplay(visView, visCam, visVol, visBox, visAxes, img2D, mode=mode) 


def update_visPlaneXY(img3D, visOrtho_XY, nz):
    
    #yxz (tif) to zyx (VisPy)
    img3D = np.transpose(img3D, (2,0,1))

    #Get Orthogonal
    # [nz, ny, nx] = img3D.shape         
    img2D = np.array([img3D[nz, :, :]])        
    
    # VisPlot: 2D XY
    visView, visCam, visVol, visBox, visAxes = visOrtho_XY 
    update_visDisplay(visView, visCam, visVol, visBox, visAxes, img2D) 

def update_visPlaneYZ(img3D, visOrtho_YZ, nx):
    
    #yxz (tif) to zyx (VisPy)
    img3D = np.transpose(img3D, (2,0,1))

    #Get Orthogonal
    # [nz, ny, nx] = img3D.shape         
    img2D = np.array([img3D[:, :, nx]])        
    
    # VisPlot: 2D XY
    visView, visCam, visVol, visBox, visAxes = visOrtho_YZ 
    update_visDisplay(visView, visCam, visVol, visBox, visAxes, img2D) 

def update_visPlaneXZ(img3D, visOrtho_XZ, nz):
    
    #yxz (tif) to zyx (VisPy)
    img3D = np.transpose(img3D, (2,0,1))

    #Get Orthogonal
    # [nz, ny, nx] = img3D.shape         
    img2D = np.array([img3D[nz, :, :]])        
    
    # VisPlot: 2D XY
    visView, visCam, visVol, visBox, visAxes = visOrtho_XZ 
    update_visDisplay(visView, visCam, visVol, visBox, visAxes, img2D) 











# =============================================================================
# Extra 
# =============================================================================
def plot_visVol(visViewBox, img3D):   
    visVol = scene.visuals.Volume(img3D, parent=visViewBox.scene)
    visVol.transform = STTransform(translate=(0,0,0), scale=(1, 1, 1))
    # visVol.set_gl_state('translucent', depth_test=False)
    return visVol   
    

def plot_visBox(visViewBox, dimZYX, xyz=np.array([0,0,0])):
    x, y, z = xyz
    [sz, sy, sx] = dimZYX
        
    visBox = scene.visuals.Box(width=sx, height=sz, depth=sy,
                                color=None, #(1, 1, 1, 0)
                                # edge_color=(0, 1, 0, 1),
                                edge_color=(1, 1, 1, 0.2),
                                parent=visViewBox.scene) 
    # [sz, sy, sx] = dimZYX//2 + 1
    # visBox.transform = STTransform(translate=(sz/2 + z, sx/2 + x, sy/2 + y), scale=(1, 1, 1))
    visBox.transform = STTransform(translate=(sx/2 + x , sy/2 + y, sz/2 + z), scale=(1, 1, 1))
    # visBox.transform = STTransform(translate=(sx + x , sy + y, sz + z), scale=(1, 1, 1))
    visBox.set_gl_state('translucent', depth_test=False)  
    return visBox


def plot_visXYZAxis(visViewBox, traslate=np.zeros(3), scale=np.ones(3) , rotate=np.identity(3), isText=True, dpi=100):
    # print()
    # print('Start: plot_visXYZAxis()')
    
    # if len(traslate)==0:
    #     traslate = np.zeros(3)
    # if len(scale)==0:
    #     scale = np.ones(3)        
    # if len(rotate)==0:
    #     rotate = np.identity(3)

    # Plot XYZAxis 
    visXYZAxis = scene.visuals.XYZAxis(width=2, parent=visViewBox.scene)
    
    # Create the Affine Matrix : Traslation, Rotation, Scaling
    [x, y, z] = traslate
    M = np.concatenate((rotate.T, np.asarray([[x, y, z]])), axis=0)
    M = np.concatenate((M, np.asarray([[0, 0, 0, 1]]).T), axis=1)
    M[0] = scale[0]*M[0]
    M[1] = scale[1]*M[1]
    M[2] = scale[2]*M[2]    
    
    # print()
    # print("Matrix=\n", M)  

    mt = MatrixTransform(matrix=M)
    visXYZAxis.transform = mt
    
    # isText = False
    if isText==True:
        xyz = scale.copy()
        
        # pad = 0.25*(xyz[np.nonzero(xyz)]).min()
        # pad = 0.25*(xyz.mean())
        # pad = np.round(0.25*(np.sort(xyz)[1:])).astype(int)
        # pad = 0.25*(np.sort(xyz)[1:]).mean()
        # pad = 0.25*(xyz.min())
        pad = 0.15*(xyz[xyz>1]).min()
        
        # print('')
        # print('xyz \n', xyz)
        # print('pad \n', pad)
        
        zyxZ = [xyz[0] + pad, 0, 0]
        label = 'X'
        plot_Text(visViewBox, zyxZ, label, c='r', dpi=dpi)
        
        zyxY = [0, xyz[1] + pad, 0]
        label = 'Y'
        plot_Text(visViewBox, zyxY, label, c='g', dpi=dpi)    
        
        zyxX = [0, 0, xyz[2] + pad ]
        label = 'Z'
        plot_Text(visViewBox, zyxX, label, c='b', dpi=dpi)  

    # print()
    # print('Stop: plot_visXYZAxis()')
    return visXYZAxis
    
    
    # print()
    # print("plot_visXYZAxis")
    # print()
    # print("traslate=", traslate)
    # print()
    # print("scale=", scale)
    # print()
    # print("rotate\n=", rotate)
    # print()
    # print("Matrix=\n", M)  
    
    
    # zyx = (scale//2)[[2,1,0]]
        
    #     # zyxZ = zyx.copy()
    #     # label = 'Z'
    #     # self.plot_Text(view, zyxZ, label)
        
    #     zyxY = [[-0.25*(zyx[2]), zyx[1] , -0.25*(zyx[2])]]
    #     label = 'Y'
    #     print()
    #     print('type')
    #     print(zyx)
    #     print(zyxY)
    #     print(type(zyxY))
    #     # sys.exit()
    #     self.plot_Text(view, zyxY, label, c='g', dpi=dpi)   
    
# =============================================================================
# 
# =============================================================================



if __name__== '__main__':
    pass

# =============================================================================
# Drft
# =============================================================================

    
    
    
    
    
    