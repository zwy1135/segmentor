# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 15:22:07 2014

@author: wy
"""
from __future__ import print_function

import numpy as np
import networkx as nx





def computeSimilarity(feature1,feature2):
    assert len(feature1)==len(feature2)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    #cosine similarity
    similarity = (sum(feature1*feature2))/((sum(feature1**2)**0.5)*(sum(feature2**2)**0.5))
    #similarity = sum((feature1-feature2)**2)
    
    return similarity
    
def computeWeight(zeroProba,oneProba):
    zeroWeight = zeroProba / (zeroProba+oneProba)
    oneWeight = oneProba / (zeroProba+oneProba)
    
    return zeroWeight,oneWeight
    
def computeConectivity(faces):
    print("computing connectivity.")
    Map = {}
    size = len(faces)
    conectivity = set()
    for i in range(size-1):
        for vertex in faces[i]:
            Map.setdefault(vertex,set())
            Map[vertex].add(i)
    print("vertex map done.")
    for faceSet in Map.values():
        faceSet = list(faceSet)
        for i in range(len(faceSet)):
            for j in range(i,len(faceSet)):
                if i==j:
                    continue
                count = 0
                for vertex in faces[faceSet[i]]:
                    if vertex in faces[faceSet[j]]:
                        count += 1
                    if count >1:
                        conectivity.add((faceSet[i],faceSet[j]))
                        continue
    return conectivity
    
def buildDiGraph(faces,features,proba):
    DG = nx.DiGraph()
    
    graphSize = len(faces)+2
    source = graphSize -2
    sink = graphSize -1
    
    DG.add_nodes_from(range(graphSize))
    
    for i in range(graphSize-2):
        zeroWeight,oneWeight = computeWeight(proba[i][0],proba[i][1])
        DG.add_edge(source,i,weight=zeroWeight)
        DG.add_edge(i,sink,weight = oneWeight)
        
    conectivity = computeConectivity(faces)
    
    for u,v in conectivity:
        similarity = computeSimilarity(features[u],features[v])
        DG.add_edge(u,v,weight = similarity)
        DG.add_edge(v,u,weight = similarity)
        
    return DG,graphSize
    
def cutAndLabel(graph,size):
    source = size -2
    sink = size - 1
    
    cutValue,partition = nx.minimum_cut(graph,source,sink,capacity="weight")
    zeroIndex,oneIndex = partition
    
    if source in zeroIndex:
        zeroIndex.remove(source)
    if sink in oneIndex:
        oneIndex.remove(sink)
    result = np.zeros(size-2)
    result[list(zeroIndex)] = 0
    result[list(oneIndex)] = 1
    
    return result
    
    

