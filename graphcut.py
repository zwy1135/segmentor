# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 15:22:07 2014

@author: wy
"""
from __future__ import print_function

from copy import deepcopy

import numpy as np
import networkx as nx






def computeSimilarity(feature1,feature2):
    assert len(feature1)==len(feature2)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    #cosine similarity
    similarity = 0.6 * ((sum(feature1*feature2))/((sum(feature1**2)**0.5)*(sum(feature2**2)**0.5))) ** 2
    #similarity = sum((feature1-feature2)**2)
    
    return similarity
    
def computeWeight(trueProba):
    falseWeight = 1 - trueProba
    trueWeight = trueProba
    
    return falseWeight,trueWeight
    
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
    graphList = []
    
    graphSize = len(faces)+2
    source = graphSize -2
    sink = graphSize -1
    
    DG.add_nodes_from(range(graphSize))
    
    conectivity = computeConectivity(faces)
    
    for u,v in conectivity:
        similarity = computeSimilarity(features[u],features[v])
        DG.add_edge(u,v,weight = similarity)
        DG.add_edge(v,u,weight = similarity)
    
    
    for label in range(1,len(proba[0])):
        graph = deepcopy(DG)
        for i in range(graphSize-2):
            falseWeight,trueWeight = computeWeight(proba[i][label])
            graph.add_edge(source,i,weight=falseWeight)
            graph.add_edge(i,sink,weight = trueWeight)
            
        graphList.append(graph)
        
    
        
    return graphList,graphSize
    
def cutAndLabel(faces,features,proba):
    graphList,size = buildDiGraph(faces,features,proba)    
    
    source = size -2
    sink = size - 1
    
    result = np.zeros(size-2)    
    
    for i in range(len(graphList)):
        cutValue,partition = nx.minimum_cut(graphList[i],source,sink,capacity="weight")
        falseIndex,trueIndex = partition
        
        if sink in trueIndex:
            trueIndex.remove(sink)
        
        result[list(trueIndex)] = i+1
    
    return result
    
    

