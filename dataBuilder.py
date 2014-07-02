# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 19:58:54 2014

@author: wy
"""

import numpy as np
import os


modelPath = './objdata'
featurePath = './feature'
labelPath = './seg'

def fileMapping(path):
    fileMap = {}    
    
    filelist = os.listdir(path)
    for f in filelist:
        key = f.strip().split('.',1)[0]
        fileMap.setdefault(key,[])
        fileMap[key].append(f)
    for key in fileMap:
        fileMap[key].sort()
        
    return fileMap
    
def featureMapping(fileMap = fileMapping(featurePath)):
    features = {}
    for key in fileMap.keys():
        feature = [np.loadtxt(f) for f in fileMap[key]]
        features[key] = np.array(feature).T
    return features
    
def faceMapping(modelPath = modelPath):
    faceMap = {}
    modelList = os.listdir(modelPath)
    for m in modelList:
        key = m.strip().split('.',1)[0]
        faces = []
        for line in file(m):
            if line.startswith('v'):
                continue
            if line.startswith('f'):
                faces.append(map(int,line.strip().split(' ')[-3:]))
        faces = np.array(faces)-1
        faceMap[key] = faces
    return faceMap
    
def getFeatureOfFace(faceMap = faceMapping(),featureMap = featureMapping()):
    faceFeature = {}
    for key in faceMap:
        faces = faceMap[key]
        feature = featureMap[key]
        buf = [np.average(feature[f],axis = 0) for f in faces]
        buf = np.array(f)
        faceFeature[key] = buf
    return faceFeature
    
def labelMapping(labelPath = labelPath):
    fileList = os.listdir(labelPath)
    labelMap = {}
    for f in fileList:
        key = f.strip().split('.',1)
        labelMap[key] = np.loadt(f).flatten()
        
    return labelMap
    
def dataAndLabel(faceFeature = getFeatureOfFace(),labelMap = labelMapping()):
    keys = labelMap.keys()
    data = []
    label = []
    for k in keys:
        if not k in faceFeature:
            continue
        data.append(faceFeature[k])
        label.append(labelMap[k])
    data = np.array(data)
    label = np.array(label)
    return (data,label)
    

    

    
if __name__ == '__main__':
    print dataAndLabel()
    

