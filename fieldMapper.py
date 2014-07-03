# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 15:37:37 2014

@author: zeng
"""

import os

from interfaces import mapping

def moveFile(src,des):
    usage = 'mv %s %s'%(src,des)
    print 'moving : %s'%usage
    os.system(usage)

def getMaps(objPath = './objdata',
            fieldPath = './feature',
            config = './config/config.conf',
            field = 'normals'):
    objlist = os.listdir(objPath)
    for filename in objlist:
        fullFilename = os.path.join(objPath,filename)        
        mapping(fullFilename,config)
    srcFile = os.path.join(objPath,'*.map')
    moveFile(srcFile,fieldPath)
    
if __name__=='__main__':
    getMaps()
    