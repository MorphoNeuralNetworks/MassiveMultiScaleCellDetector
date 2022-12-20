# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:23:03 2021

@author: aarias
"""


#GUI 
from PyQt5 import QtWidgets, QtCore,  QtGui


#Path Management
from pathlib import Path

import pandas as pd

from IO.Files.FileManager import createFolder

def open_QTDialogToGetpathFolderfromFile(self, label='Select an Image File', verbose=False):
    pathFile, selectedFilter = QtWidgets.QFileDialog.getOpenFileName(self, label)
    pathFile  = Path(pathFile)
    fileName   = pathFile.name
    folderName = pathFile.parent.name 
    pathFolder = pathFile.parent
    
    if verbose==True:
        print()
        print('pathFolder \n', pathFolder)
        print('folderName \n', folderName)
        print('fileName \n', fileName) 
        print('selectedFilter \n', selectedFilter)
        
    return pathFolder

def open_QTDialogToGetpathFolderfromFolder(self, label='Select a Folder', verbose=False):
    pathFolder = QtWidgets.QFileDialog.getExistingDirectory(self, label)
    pathFolder  = Path(pathFolder)

    if verbose==True:
        print()
        print('pathFolder \n', pathFolder)
 
        
    return pathFolder
# =============================================================================
#          
# =============================================================================
class ScrollBarPaired(QtCore.QObject):
    def __init__(self, scrollBarMin, scrollBarMax, parent=None):
        # super(ScrollBarPaired, self).__init__(parent)
        self.scrollBarMin = scrollBarMin
        self.scrollBarMax = scrollBarMax
        self.scrollBarMin.valueChanged.connect(self.event_ScrollBarMin)
        print('heyyyye')
    
    def event_ScrollBarMin(self):
        print('hola')
        
        
# class ScrollBarPaired(QtWidgets.QScrollArea, QtWidgets.QScrollArea):
#   def __init__( self, parent):
#       super(ScrollBarPaired, self).__init__(parent)

      # self.pushButton = QtGui.QPushButton(self)
      
# class ScrollBarPaired():

#     def __init__(self, scrollBarMin, scrollBarMax):
#         super(ScrollBarPaired, self).__init__()  
#         self.scrollBarMin = scrollBarMin
#         self.scrollBarMax = scrollBarMax
        
#         self.scrollBarMin.valueChanged.connect(self.event_ScrollBarMin)
#         self.scrollBarMax.valueChanged.connect(self.event_ScrollBarMax)
        
#     def event_ScrollBarMin(self, value):
#         print('hoallll')
#         Vmin = value
#         Vmax = self.ui.horizontalScrollBarMax.value()
#         if Vmin>Vmax:
#             self.ui.scrollBarMax.setValue(Vmin)            
#         # self.ui.label_Vmin.setText(str(Vmin))

#     def event_ScrollBarMax(self, value):
#         Vmin = self.ui.horizontalScrollBarMin.value() 
#         Vmax = value
#         if Vmin>Vmax:
#             self.ui.horizontalScrollBarMin.setValue(Vmax)            
#         # self.ui.label_Vmax.setText(str(Vmax))
# =============================================================================
# 
# =============================================================================
# Step 1: Create a worker class
class Worker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    results   = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal()
    
    
    # resultsReady = pyqtSignal(np.ndarray, int, int)
    # resultsReady = pyqtSignal()
    # progress = pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        super(QtCore.QObject, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        
        
    def run(self):        
        print()
        print('----------------------------')
        print('.....Long-Running Task......')
        print('----------------------------')
        
        #?????
        res = self.fn(*self.args, **self.kwargs)
        # self.results.emit(res)
        self.finished.emit()


# =============================================================================
# 
# =============================================================================
# from natsort import index_natsorted, order_by_index
class PandasModel(QtCore.QAbstractTableModel):
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(PandasModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.pyqtProperty(pd.DataFrame, fget=dataFrame, fset=setDataFrame)

    @QtCore.pyqtSlot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() \
            and 0 <= index.column() < self.columnCount()):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == PandasModel.ValueRole:
            return val
        if role == PandasModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            PandasModel.DtypeRole: b'dtype',
            PandasModel.ValueRole: b'value'
        }
        return roles
 
    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()
# =============================================================================
#       
# =============================================================================

if __name__== '__main__':
    pass    
         















