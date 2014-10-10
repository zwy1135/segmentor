# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 19:58:54 2014

@author: wy
"""
from __future__ import print_function
import os


import numpy as np



modelPath = './objdata'
featurePath = './feature'
labelPath = './seg'

def fileMapping(path):
    print('Mapping dir %s.'%path)
    fileMap = {}    
    
    filelist = os.listdir(path)
    for f in filelist:
        key = f.strip().split('.',1)[0]
        fileMap.setdefault(key,[])
        fileMap[key].append(os.path.join(path,f))
    for key in fileMap:
        fileMap[key].sort()
        
    return fileMap
    
def featureMapping(fileMap = fileMapping(featurePath)):
    print('Mapping feature.')
    features = {}
    for key in fileMap.keys():
        feature = [np.loadtxt(f) for f in fileMap[key]]
        features[key] = np.array(feature).T
    return features
    
def faceMapping(modelPath = modelPath):
    print('Mapping face dir %s.'%modelPath)
    vertexMap = {}
    faceMap = {}
    modelList = os.listdir(modelPath)
    for m in modelList:
        key = m.strip().split('.',1)[0]
        faces = []
        vertex = []
        with open(os.path.join(modelPath,m)) as f:
            for line in f:
                if line.startswith('v'):
                    vertex.append(list(map(float,line.strip().split(' ')[-3:])))
                if line.startswith('f'):
                    faces.append(list(map(int,line.strip().split(' ')[-3:])))
        faces = np.array(faces)-1
        vertex = np.array(vertex)
        faceMap[key] = faces
        vertexMap[key] = vertex
    return faceMap,vertexMap
    
def getFeatureOfFace(fv = faceMapping(),featureMap = featureMapping()):
    print('getting feature of face')
    (faceMap,vertexMap) = fv
    faceFeature = {}
    for key in faceMap:
        vertex = vertexMap[key]
        faces = faceMap[key]
        feature = featureMap[key]
        faceFeatureEach = np.zeros((feature.shape[0],feature.shape[1]+3))
        faceFeatureEach[:,:3] = vertex
        faceFeatureEach[:,3:] = feature
        buf = [np.average(faceFeatureEach[f],axis = 0) for f in faces]
        buf = np.array(buf)
        faceFeature[key] = buf
    return faceFeature
    
def labelMapping(labelPath = labelPath):
    print('Mapping labels dir %s.'%labelPath)
    fileList = os.listdir(labelPath)
    labelMap = {}
    for f in fileList:
        key = f.strip().split('.',1)[0]
        labelMap[key] = np.loadtxt(os.path.join(labelPath,f)).flatten()
        
    return labelMap
    
def dataAndLabel(faceFeature = getFeatureOfFace(),labelMap = labelMapping()):
    print('Getting data and label.')
    keys = labelMap.keys()
    data = []
    label = []
    for k in keys:
        if not k in faceFeature:
            continue
        data += list(faceFeature[k])
        label += list(labelMap[k])
    data = np.array(data)
    label = np.array(label)
    return (data,label)
    

    

    
if __name__ == '__main__':
    data,label = dataAndLabel()
    print(data,label)
    print(len(data),len(label))
    

