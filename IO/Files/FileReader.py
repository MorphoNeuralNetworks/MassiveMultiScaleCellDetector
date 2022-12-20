# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:05:42 2020

@author: pc
"""

import glob

def get_pathFiles(folderPath, fileNamePattern):    

    pathFiles = (glob.glob(folderPath + fileNamePattern))

    return pathFiles
    

# def read_CSV(df, folderPath, fileName, sep=','):
    
#     #Saving the Table
#     fileExtension = '.csv'
#     filePath = os.path.join(folderPath, fileName + fileExtension)  
#     #df.to_csv(filePath, sep=';', encoding='utf-8', index=True)  
#     df.to_csv(filePath, sep=',', encoding='utf-8', index=True) 
    
#     return df
    
if __name__== '__main__':
    pass   
