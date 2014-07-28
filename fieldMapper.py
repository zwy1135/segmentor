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
            field = 'c0-g-k1,c0-g-k2,H0,K0,SI0,lC0,DLP0,WM0,DI0,SIH0,SH0'):
    objlist = os.listdir(objPath)
    for filename in objlist:
        fullFilename = os.path.join(objPath,filename)        
        mapping(fullFilename,config,field)
    srcFile = os.path.join(objPath,'*.map')
    moveFile(srcFile,fieldPath)
    
if __name__=='__main__':
    getMaps('./testing')
    
