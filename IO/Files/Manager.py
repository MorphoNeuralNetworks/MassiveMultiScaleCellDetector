# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:43:54 2020

@author: pc
"""
import os 
import shutil
import time
import stat

def createFolder(path):
    #If the Path already exist remove all the content
    #If the Path not exist create Folder  
#    path  = Path(path)
    print(path)
    if os.path.exists(path): 
        print('Remove Folder Content...')
#        shutil.rmtree(path)
        rmtree(path)
#        time.sleep(5.0)
        
    #This dealy is required to avoid the following Error
    #PermissionError: [WinError 5] Access is denied
    #It seems that the OS requires time to finish the above operation
#    time.sleep(0.000000001)
    time.sleep(0.00001)
    
    if not os.path.exists(path):
        print('Create Folder...')
        os.makedirs(path)



#Custumize Shutil.rmtree
def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top) 
    
    
if __name__== '__main__':
    pass