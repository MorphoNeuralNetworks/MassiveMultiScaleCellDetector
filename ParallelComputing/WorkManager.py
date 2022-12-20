# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 01:12:23 2020

@author: aarias
"""
import numpy as np

import concurrent.futures

from IO.Files.FileWriter import save_Figure
from IO.Files.FileManager import createFolder

import time

from tqdm import tqdm

import sys

import matplotlib
# matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
# Ensure using PyQt5 backend
# matplotlib.use('QT5Agg')


#==============================================================================
# Repeats
#==============================================================================
#        myArgs = [imgPaths, itertools.repeat([x0, x1, y0, y1], len(imgPaths))]
#        myArgs = [imgPaths, [[x0, x1, y0, y1] for i in range(len(imgPaths))]]
#==============================================================================
# 
#==============================================================================
def multithreading(func, args, workers):
    t0 = time.time()
    args.append([t0 for i in range(len(args[0]))])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, *args)
    return list(res)


def multiprocessing(func, args, workers):
    t0 = time.time()
    args.append([t0 for i in range(len(args[0]))])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, *args)
    return list(res)

#==============================================================================
# 
#==============================================================================
def parallelComputing(func, args, nProcesses=1, nThreads=1):
#    print('')
#    print('Launch Main Parallel')   
    t0 = time.time()
    args.append([t0 for i in range(len(args[0]))])
    
    m = []
    for i in args:
        m.append(np.array_split(i, nProcesses))
    args = list(zip(*m))
  
    func = [func for i in range(len(args))]
    nThreads = [nThreads for i in range(len(args))]
    
    
#    print(func)
#    print(args)
#    print(nProcesses)
#    print(nThreads)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=nProcesses) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=nProcesses) as executor:        
        res = executor.map(run_threads, func, args, nThreads)    
        # res = list(tqdm(executor.map(run_threads, func, args, nThreads), total=len(nThreads))) 
    return list(res)   
 
# list(tqdm(executor.map(f, my_iter), total=len(my_iter)))

def run_threads(func, args, workers):
#    print('')
#    print('From Process Launch Threads: Start')
#
#    print(func)
#    print(args)
#    print(workers) 
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
    # with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # res = executor.map(func, *args) 
        res = list(tqdm(executor.map(func, *args), total=len(args[0]))) # ???
    return list(res)    



#==============================================================================
# 
#==============================================================================
def mySum(a,b,t0):    
    start = time.time() - t0
    time.sleep(0.25)
    c = a + b
    stop = time.time() - t0
    return c, start, stop


def mySum5(a,b,t0): 
    print(a)
    print(b)
    print(t0)
    
    M1 = [[1, 2],[3,4]]
    M2 = [[1,2,3], [4,5,6],[7,8,9]]
    M = [M1, M2]
    return M
        
def mySum6(t0, *args):
    print (args)
    a, b, T0 = args
    M1 = [[1, 2],[3,4]]
    M2 = [[1,2,3], [4,5,6],[7,8,9]]
    M = [M1, M2]
    return M  
    
    
    
#==============================================================================
#     
#==============================================================================
def plot_TaskRunTimes(ax, start, stop, bar_color, label=None):       
    widths   = stop - start
    xcenters = start + widths/2.0  
    ycenters = np.arange(1, len(start)+1)
    ax.barh(ycenters, widths, left=start, color=bar_color, align='center', alpha=0.5, label=label)
    for xcenter, ycenter, width in zip(xcenters, ycenters, widths):
        ax.text(xcenter, ycenter, '{:0.2f}'.format(width), ha='center', va='center', color='k')
  
    return stop[-1]-start[0]

def plot_ComputingPerformance(df, computation_labels):
    
    #colorMaps: 'RdYlGn', 'hsv', 'Pastel1','Paired'
    category_names = np.unique(df['compID'])
    n_catergories = category_names.shape[0]
    category_colors = plt.get_cmap('Paired')(np.linspace(0, 1, n_catergories))
    
    nTasks = np.unique(df['taskID']).shape[0]
#    nProcesses = np.unique(df['processID']).shape[0] 
#    nThreads = np.unique(df['threadID']).shape[0] 
    

    stop = df['stop'].max()
    inches_per_second = 0.28
#    inches_per_second = 2.28
    # print(stop+1)
    # print((stop+1)*inches_per_second)
    # print(nTasks*1)
    # jaja
    fig, ax = plt.subplots(figsize=((stop+1)*inches_per_second, nTasks*1)) 
#    fig, ax = plt.subplots()
    for i, (colname, color) in enumerate(zip(computation_labels, category_colors)):
        #Plot bars
        dfSub = df[(df['compID']==category_names[i])]
        ycenters = np.asarray(dfSub['taskID'])
        starts = dfSub['start'] 
        widths = dfSub['width']   
        ax.barh(ycenters, widths, left=starts, height=0.5, align="center", label=colname, color=color, alpha= 0.75)

        #Plot texts
        xcenters = starts + widths / 2
        text_color = 'k'
        for j, (x, c) in enumerate(zip(xcenters, widths)):
            #Plot numerical value of the time if greater than 1 second
            if c>=1.0:
                myText = '{:0.1f}'.format(c)
                ax.text(x, ycenters[j], myText, ha='center', va='center',color=text_color)
        

    #For loop for the task
    for task in range(1,nTasks+1):           
        dfsub = df[(df['taskID']==task)]        
        process_ID = dfsub['processID'].iloc[0]
        thread_ID = dfsub['threadID'].iloc[0]
        myText = 'p_Id:' + '{:0.0f}'.format(process_ID) + ', t_Id:' + '{:0.0f}'.format(thread_ID)
#        x_center = (dfsub['stop'].max() - dfsub['start'].min())/2.0 + dfsub['start'].min()
#        ax.text(x_center, task + 0.30, myText, ha='left', va='bottom',color=text_color)
        
        x_left = dfsub['start'].min()
        ax.text(x_left, task + 0.30, myText, ha='left', va='bottom',color=text_color)
    #Plot Total
    ycenter = 0.0
    start = df['start'].min()
    stop = df['stop'].max()
    width = stop - start
    colname = 'Total'    
    ax.barh(ycenter, width, left=start, height=0.5, align="center", label=colname, color='0.25', alpha=0.5)
    
    xcenter = start + width / 2
    myText = '{:0.1f}'.format(width)
    ax.text(xcenter, ycenter, myText, ha='center', va='center',color=text_color)

#    #Title Settings 
#    figure_title = ('nWorkers='+ str(nProcesses*nThreads) + ', nProcesses='+ str(nProcesses) + ', nThreads='+ str(nThreads))
#    ax.set_title(figure_title, fontsize=12)  
    
    #Legend Settings     
    ax.legend(loc='upper left',
              fontsize='small',
              bbox_to_anchor=(1.05, 1, 1., 0.),
              borderaxespad=0,              
              mode=None) #mode=None ,'expand'   
              
    #Axis Settings:    
    ax.set_yticks(np.arange(0, nTasks + 1)) 
    ax.set_xlim(0, ax.axes.get_xlim()[1])
    ax.set_ylim(-0.5, nTasks + 1)   
#    ax.set_ylabel('Task') 
#    ax.set_xlabel('Time [seconds]')        
    
#    plt.show()
    
#    createFolder(str(pathFolder), remove=False)
#    
#    fileName = (
#                str(scannerSize_ani[0]) + 'x' + str(scannerSize_ani[1]) + 'x' + str(scannerSize_ani[2]) +
#                '_nW_' + str(nProcesses*nThreads) + 
#                '_nP_' + str(nProcesses) +
#                '_nT_' + str(nThreads) )
#    save_Figure(fig, pathFolder, fileName)
    
    return fig, ax
# =============================================================================
# 
# =============================================================================

# =============================================================================
# 
# =============================================================================
def plot_ComputingPerformance2(df, computation_labels=None, sortedBy='taskID', skipStart=True):
    
    # =============================================================================
    #   Select Plot Settings
    # =============================================================================
    
    #Selecting the ColorMap of the Operations Lables (ColorMaps Options: 'RdYlGn', 'hsv', 'Pastel1','Paired')
    category_names = np.unique(df['compID'])
    n_catergories = category_names.shape[0] 
    category_colors = plt.get_cmap('Paired')(np.linspace(0, 1, n_catergories))
    
    #if not given... Selecting the Operation Tags
    if (computation_labels==None) or (len(list(computation_labels))!=n_catergories):
        computation_labels = list(category_names)
    
    #Selecting sortedBy Options; "taskID" or "workerID" or "unknown"
    nTasks = np.unique(df['taskID']).shape[0]
    if (sortedBy=='taskID') or (sortedBy==None):
        print()
        print('plot_ComputingPerformance: sortedBy taskID')        
    elif (sortedBy=='workerID'): 
        print()
        print('plot_ComputingPerformance: sortedBy workerID')
        df['taskID_bk'] = df['taskID']
        
        a = df['threadID']
        _, ix = np.unique(a, return_index=True)
        threadsID = a[np.sort(ix)]
        
        nWorkers = threadsID.shape[0]
        nTasks = nWorkers
        # Overwrite the taskIDs with the workerIDs
        for i in range(0, nWorkers):      
            maskBool = (df['threadID']==threadsID.iloc[i]) 
            b = df['taskID'].values
            b[maskBool] = i + 1
            df['taskID'] = b              
    else:
        print()
        print('plot_ComputingPerformance: sortedBy unknown')
          
    #Selecting the minimum "comnputing time" to be plotted in the bars as a string
    maskBool = df['width']>0    
    # dt_min = df['width'][maskBool].min()
    dt_min = df['width'][maskBool].mean()
      
    #Selecting the Figure Dimensions
    start = df['start'].min()
    stop = df['stop'].max()
    width = stop - start
    inches_per_second = 0.28
    # inches_per_second = 0.5*(1/dt_min)    
    fig, ax = plt.subplots(figsize=((width)*inches_per_second, nTasks*1)) 
    
    # ???
    # fig, ax = plt.subplots(figsize=((stop+1)*inches_per_second, nTasks*1))
    
    #????
    # fig = Figure(figsize=((width)*inches_per_second, nTasks*1))
    # ax = fig.add_subplot(111)
    
    
    # fig = self.ui.plotWidget.canvas.fig
    # fig.clf()
    # ax = fig.subplots(1,1)
    # =============================================================================
    #   Start Ploting the "Computing Performance" Graph     
    # =============================================================================    
    
    #Ploting Loop: each loop plots the bars of each operation
    for i, (colname, color) in enumerate(zip(computation_labels, category_colors)):
        #Plot the "Horizontal Bars" that represents the time that takes each operation
        dfSub = df[(df['compID']==category_names[i])]
        ycenters = np.asarray(dfSub['taskID'])
        starts = dfSub['start'] 
        widths = dfSub['width']   
        ax.barh(ycenters, widths, left=starts, height=0.5, align="center", label=str(colname), color=color, alpha= 0.75)

        #Plot the "texts" that represents the time that takes each operation
        xcenters = starts + widths / 2
        text_color = 'k'
        for j, (x, c) in enumerate(zip(xcenters, widths)):
            if c>=dt_min:
                myText = '{:0.1f}'.format(c)
                ax.text(x, ycenters[j], myText, ha='center', va='center',color=text_color)
     

    if (sortedBy=='taskID') or (sortedBy==None):
        #Plot the "processID and threadID" above the horizontal bars
        for task in range(1, nTasks+1):           
            dfsub = df[(df['taskID']==task)]        
            process_ID = dfsub['processID'].iloc[0]
            thread_ID = dfsub['threadID'].iloc[0]
            myText = 'p_Id:' + '{:0.0f}'.format(process_ID) + ', t_Id:' + '{:0.0f}'.format(thread_ID)       
            x_left = dfsub['start'].min()
            ax.text(x_left, task + 0.30, myText, ha='left', va='bottom',color=text_color)
    
    elif (sortedBy=='workerID'): 
        #Plot the "taskID" above the horizontal bars 
        # taskID_bk = df['taskID_bk'].unique()
        for i in range(0, nWorkers):  
            dfsub = df[(df['threadID']==threadsID.iloc[i])]  
            taskID_bk = dfsub['taskID_bk'].unique()
            for j in range(0, taskID_bk.shape[0]):
                dfsubsub = dfsub[(dfsub['taskID_bk']==taskID_bk[j])] 
                myText = 'task_Id:' + '{:0.0f}'.format(taskID_bk[j])       
                x_left = dfsubsub['start'].min()
                ax.text(x_left, i + 1.30, myText, ha='left', va='bottom',color=text_color)        
        
        
    #Plot the "Total Time" Horizontal Bar
    ycenter = 0.0
    colname = 'Total'    
    ax.barh(ycenter, width, left=start, height=0.5, align="center", label=colname, color='0.25', alpha=0.5)
    
    #Plot the "Total Time" text
    xcenter = start + width / 2
    myText = '{:0.1f}'.format(width) 
    ax.text(xcenter, ycenter, myText, ha='center', va='center',color=text_color)

    # =============================================================================
    #    Select Plot Lables 
    # =============================================================================
    
    #Legend Settings     
    ax.legend(loc='upper left',
              fontsize='small',
              bbox_to_anchor=(1.05, 1, 1., 0.),
              borderaxespad=0,              
              mode=None) #mode=None ,'expand'   
              
    #Axis Settings:    
    ax.set_yticks(np.arange(0, nTasks + 1)) 
    if skipStart==False:
        ax.set_xlim(0, ax.axes.get_xlim()[1])
    else:
        ax.set_xlim(start, ax.axes.get_xlim()[1])
        print()
        print('start', start)
    ax.set_ylim(-0.5, nTasks + 1) 
    ax.set_ylim(-0.5, nTasks + 1)   
    
    #Axis Labels:
    myFontSize = 20
    ax.set_ylabel(str(sortedBy), fontsize=myFontSize) 
    ax.set_xlabel('Time [seconds]', fontsize=myFontSize)   
         
    return fig, ax





if __name__== '__main__':
    from matplotlib import pyplot as plt

    a = [1,2]
    b = [5,6]    
    a = [1,2,3,4]
    b = [5,6,7,8]
    args = [a,b]
    n_Tasks = len(args[0])


#==============================================================================
# Debuguin
#==============================================================================
#    res = parallelComputing(mySum, args, nProcesses=4, nThreads=1)
#    res0 = parallelComputing(mySum5, args, nProcesses=2, nThreads=2)
#    res0 = parallelComputing(mySum, args, nProcesses=2, nThreads=2)
    res0 = parallelComputing(mySum5, args, nProcesses=2, nThreads=2)
#    res0 = parallelComputing(mySum6, args, nProcesses=1, nThreads=1)



#    print('')
#    print res0
    
    
    # res0 = filter(None, res0)  
    # M =  np.array(res0, dtype=object)    
    # M = np.concatenate(M, axis=0)
    # M1, M2 = M[:,0], M[:,1]
    
    # n1 = M1.shape[0]
    # M1 = np.concatenate(M1, axis=0)
    # M1 = M1.reshape(n1,-1)
    
    # n2 = M2.shape[0]
    # M2 = np.concatenate(M2, axis=0)
    
#    print('')
#    print res0
#    
#    print('')
#    print M1
#    print('')
#    print M1.shape
#    
#    print('')
#    print M2
#    print('')
#    print M2.shape
    
#==============================================================================
#     
#==============================================================================
#    res = parallelComputing(mySum, args=args, nProcesses=2, nThreads=2)
#    
#    res = np.array(res)
#    nx, ny, nz = res.shape
#    res = res.reshape((nx*ny), nz)          
#    [cSum, start, stop] = np.array(res).T 
#    
#    print (res) 
#    print (start)
#    print (stop)
#    
#    #Figure: Settings
#    ny, nx = 1, 1
#    m = 1.0    
#    fig, ax = plt.subplots(ny,nx)
#    graphSize = [7.0, 4.0]
#    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
#    fig.set_size_inches(graphSize)    
#    
#    #Figure: Labels
#    ax.set_title('Serial. ' + 'Tasks:'+ str(n_Tasks))
#    ax.set_xlabel("Seconds")
#    ax.set_ylabel("Tasks")
#    
#    #Figure: Plots    
#    plot_TaskRunTimes(ax, start=start, stop=stop, bar_color='b', label='IO')   
#    
#    #Settings:    
#    ax.set_yticks(np.arange(1, n_Tasks + 1)) 
#    ax.set_ylim(0, n_Tasks + 1.5)
#    ax.set_xlim(0, ax.axes.get_xlim()[1])
#              
#    ax.legend(loc='upper center', bbox_to_anchor=(0., +0.95, 1., 0.),
#              borderaxespad=0, ncol=2, mode=None) #mode='expand'
#    
#    plt.show()   


    
    
#    widths   = stop - start
#    xcenters = start + widths/2.0  
#    ycenters = np.arange(1, len(start)+1)
#    ax.barh(ycenters, widths, left=start, color=bar_color, align='center', alpha=0.5, label=label)
#    
#    ycenters = [1 , 2]
#    widths = [2, 3]
#    starts = [1, 3]
#    plt.barh(ycenters, widths, left=starts, align='center')
#==============================================================================
# 
#==============================================================================
#    a = [1,2]
#    b = [5,6]    
#    a = [1,2,3,4]
#    b = [5,6,7,8]
#    args = [a,b]
#    res = parallelComputing(mySum2, args, nProcesses=3, nThreads=1)
##    res = parallelComputing(mySum, args, nProcesses=4, nThreads=1)
##    res = parallelComputing(mySum, args, nProcesses=3, nThreads=1)
#
#    M = np.array(res, dtype=object)
#    nx, ny, nz = M.shape
#    M = M.reshape((nx*ny), nz)        
#    [cSum, dt1, dt2] = np.array(M).T 
#    
#    dt1 = np.concatenate(dt1).reshape(nx*ny,2)
#    dt1 = dt1.T
#    start, stop = dt1
#    
#    dt2 = np.concatenate(dt2).reshape(nx*ny,2)
#    dt2 = dt2.T
#    start2, stop2 = dt2
#    
#    print (res) 
#    print (start)
#    print (start)
#    
#    n_Tasks = len(args[0])
#
#    #Figure: Settings
#    ny, nx = 1, 1
#    m = 1.0    
#    fig, ax = plt.subplots(ny,nx)
#    graphSize = [7.0, 4.0]
#    graphSize = m*nx*graphSize[0], m*ny*graphSize[1]    
#    fig.set_size_inches(graphSize)    
#    
#    #Figure: Labels
#    ax.set_title('Serial. ' + 'Tasks:'+ str(n_Tasks))
#    ax.set_xlabel("Seconds")
#    ax.set_ylabel("Tasks")
#    
#    #Figure: Plots        
#    plot_TaskRunTimes(ax, start=start, stop=stop, bar_color='b', label='IO')   
#    plot_TaskRunTimes(ax, start=start2, stop=stop2, bar_color='r', label='CPU')   
# 
#    #Settings:    
#    ax.set_yticks(np.arange(1, n_Tasks + 1)) 
#    ax.set_ylim(0, n_Tasks + 1.5)
#    ax.set_xlim(0, ax.axes.get_xlim()[1])
#              
#    ax.legend(loc='upper center', bbox_to_anchor=(0., +0.95, 1., 0.),
#              borderaxespad=0, ncol=2, mode=None) #mode='expand'
#    
#    plt.show() 
    

        
        
#==============================================================================
#   draft      
#==============================================================================
#    c, start, stop = np.array(res).T
#    start = start.flatten()
#    stop = start.flatten()
#    print('')
#    print (res)
#
#    print('')
#    print ('c    =', c)
#    print('')
#    print ('start=', start)
#    print('')
#    print ('stop =', stop)         
        