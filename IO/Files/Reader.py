# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:05:42 2020

@author: pc
"""

import glob

def get_pathFiles(folderPath, fileNamePattern):    

    pathFiles = (glob.glob(folderPath + fileNamePattern))

    return pathFiles
    
    
if __name__== '__main__':
    pass   
