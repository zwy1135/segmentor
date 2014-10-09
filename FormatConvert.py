# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 10:47:57 2014

@author: zeng
"""
import os
from __future__ import print_function

def off2obj(meshIn,meshOut):
    usage = 'python off2obj.py %s %s'%(meshIn,meshOut)
    print_function('doing: %s'%usage)
    os.system(usage)
  

def convert2obj(mshPath = '~/ms/MeshsegBenchmark-1.0/data/off',
                targetPath = '../objdata/obj'):
    
    filelist = os.listdir(mshPath)
    for filename in filelist:
        fullFilename = os.path.join(mshPath,filename)
        filename1 = filename.split('.')[0]
        targetFilename = os.path.join(targetPath,filename1+'.obj')
        off2obj(fullFilename,targetFilename)
    
if __name__=='__main__':
    convert2obj('./data','./data')


