# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:50:01 2021

@author: aarias
"""

#GPU Scientific Visualization 
import vispy
from vispy import scene
from vispy.visuals.transforms import STTransform
import vispy.io as io

#Array Computations
import numpy as np
    
# =============================================================================
#     VisPy Ploting
# =============================================================================
# Plot Volume
def plot_visVol(view, img3D):   
    visVol = scene.visuals.Volume(img3D, 
                                  
                                  parent=view.scene)
    visVol.transform = STTransform(translate=(0,0,0), scale=(1, 1, 1))

#Plot Vectors
def plot_visVector(view, p1, v, k=1.0): 
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
                                    width=2,
                                    
                                    
                                    arrows=arrows,
                                    arrow_type='triangle_30',
                                    arrow_color=[1, 0, 0, 0.5], #red with transparency
                                    arrow_size=1,
                                    
                                    method='gl',
                                    parent=view.scene)
    visArrows.transform = STTransform(translate=(0,0,0), scale=(1, 1, 1))
    visArrows.set_gl_state('translucent', depth_test=False)

#Plot Dots (x,z,y)
def plot_visDots(view, v_p, R, dpi, plot_Numbers=True):        
    pos = v_p
    pos = pos.astype(np.int32)
    visDots = scene.visuals.Markers(pos=pos, 
                                    size=5, 
                                    face_color=[0, 1, 0, 0.5],
                                    parent=view.scene) 
    
    # Plot Text 
    if plot_Numbers==True:       
        ix = np.arange(0, v_p.shape[0])
        ixText = ix.astype(str)
        visText= scene.visuals.Text(pos=pos, 
                                    text=ixText ,
                                    font_size=5*dpi,
                                    parent=view.scene)   
     
# Plot Box
def plot_visBox(view, dim):
    [sx, sy, sz] = dim
    visBox = scene.visuals.Box(width=sx, height=sy, depth=sz,
                                color=None, #(1, 1, 1, 0)
                                # edge_color=(0, 1, 0, 1),
                                edge_color=(1, 1, 1, 0.2),
                                parent=view.scene)      
    visBox.transform = STTransform(translate=(sx/2, sy/2, sz/2), scale=(1, 1, 1))
    visBox.set_gl_state('translucent', depth_test=False)  
    
    
# def plot_visSphere(self, view, p, r, verbose=False):
#     if verbose== True:
#         print()
#         print("plot_visSphere")
#         print('yxz=', p)
#         print('radius=', r)
    
#     visSphere = scene.visuals.Sphere(radius=r, 
#                                      method='latitude',
#                                      color=None,                                           
#                                      edge_color=[0, 1, 0, 0.1],
#                                      parent=view.scene,)
#     visSphere.transform = STTransform(translate=(p), scale=(1, 1, 1))
#     visSphere.set_gl_state('translucent', depth_test=False)

def plot_visSphere(view, p, r, c='g', verbose=False):
    if verbose== True:
        print()
        print("plot_visSphere")
        print('yxz=', p)
        print('radius=', r)
    
    if c=='g':
        edge_color = [0, 1, 0, 0.025]
    elif c=='r':
        edge_color = [1, 0, 0, 0.025]
    elif c=='b':
        edge_color = [0, 0, 1, 0.025]  
    visSphere = scene.visuals.Sphere(radius=r, 
                                     method='latitude',
                                     color=None,                                           
                                     edge_color=edge_color,
                                     parent=view.scene,)
    visSphere.transform = STTransform(translate=(p), scale=(1, 1, 1))
    visSphere.set_gl_state('translucent', depth_test=False)
    
# =============================================================================
# 
# =============================================================================
# Set 3D Camera
def set_visCam(view, imgDim, fov=60.0, camType='Turntable'):       
    [sx, sy, sz] = imgDim
    if camType=='Fly':            
        cam = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name='Fly')
    elif camType=='Turntable': 
        cam = scene.cameras.TurntableCamera(parent=view.scene, fov=fov, name='Turntable')
    elif camType=='Arcball':
        cam = scene.cameras.ArcballCamera(parent=view.scene, fov=fov, name='Arcball')
    else:
        print()   
        
    # Insert Cam in View
    view.camera = cam
    
    # view.camera.transform = STTransform(translate=(sx/2, sy/2, sz/2), scale=(1, 1, 1))
 
    # canvas.show()
    # view.camera.set_range()