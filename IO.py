# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:29:08 2015

@author: wy
"""
from __future__ import print_function
import os

import numpy as np



def load_model(path, name):
    if not name.endswith("obj"):
        name += ".obj"
        
    vertexes = []
    faces = []
        
    with open(os.path.join(path, name)) as f:
        print("loading %s" % name)
        for line in f:
            if line.startswith('v'):
                vertexes.append(list(map(float, line.strip().split(' ')[-3:])))
            if line.startswith('f'):
                faces.append(list(map(int, line.strip().split(' ')[-3:])))
        
    faces = np.array(faces) - 1
    vertexes = np.array(vertexes)
        
    return vertexes, faces
    
 
featureDict = {}

def build_feature_dict(path):
    global featureDict
    featureList = os.listdir(path)
    for f in featureList:
        key = f.strip().split(".")[0]
        featureDict.setdefault(key, set())
        featureDict[key].add(f) 
        

        
   
def load_feature(path, name):
    if not name in featureDict:
        build_feature_dict(path)
        
    if not name in featureDict:
        raise FileNotFoundError("feature of %s not found"%name)
        
    fileList = [os.path.join(path, f) for f in featureDict[name]]
    fileList.sort() 
    
    feature = [np.loadtxt(f) for f in fileList]
    feature = np.array(feature).T
    
    return feature
    
def load_labels(path, name):
    if not name.endswith("seg"):
        name += ".seg"
        
    return np.loadtxt(os.path.join(path, name))
            




   