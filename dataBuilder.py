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
    print 'Mapping dir %s.'%path
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
    print 'Mapping feature.'
    features = {}
    for key in fileMap.keys():
        feature = [np.loadtxt(f) for f in fileMap[key]]
        features[key] = np.array(feature).T
    return features
    
def faceMapping(modelPath = modelPath):
    print 'Mapping face dir %s.'%modelPath
    faceMap = {}
    modelList = os.listdir(modelPath)
    for m in modelList:
        key = m.strip().split('.',1)[0]
        faces = []
        for line in file(os.path.join(modelPath,m)):
            if line.startswith('v'):
                continue
            if line.startswith('f'):
                faces.append(map(int,line.strip().split(' ')[-3:]))
        faces = np.array(faces)-1
        faceMap[key] = faces
    return faceMap
    
def getFeatureOfFace(faceMap = faceMapping(),featureMap = featureMapping()):
    print 'getting feature of face'
    faceFeature = {}
    for key in faceMap:
        faces = faceMap[key]
        feature = featureMap[key]
        buf = [np.average(feature[f],axis = 0) for f in faces]
        buf = np.array(buf)
        faceFeature[key] = buf
    return faceFeature
    
def labelMapping(labelPath = labelPath):
    print 'Mapping labels dir %s.'%labelPath
    fileList = os.listdir(labelPath)
    labelMap = {}
    for f in fileList:
        key = f.strip().split('.',1)[0]
        labelMap[key] = np.loadtxt(os.path.join(labelPath,f)).flatten()
        
    return labelMap
    
def dataAndLabel(faceFeature = getFeatureOfFace(),labelMap = labelMapping()):
    print 'Getting data and label.'
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
    print data,label
    print len(data),len(label)
    
