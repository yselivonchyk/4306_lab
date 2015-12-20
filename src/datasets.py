'''
Created on 10.07.2014

@author: mkamp
'''

import collections as cl
import os

import arff
import numpy as np
#import sparse_vector
from synthetic import RapidlyDriftingDisjunction, BshoutyLongModel


SUPPORTED_FILE_TYPES = [".data", ".arff", ".csv"]


class Datasets:
    def __init__(self):
        self.datapath = os.path.dirname(os.path.abspath(__file__))+"/"
        print self.datapath
        self.datasets = cl.defaultdict(list)    
        self.datasetInfo = {}
        for stFile in os.listdir(self.datapath+"classification"):
            for i in xrange(len(SUPPORTED_FILE_TYPES)):
                if stFile.endswith(SUPPORTED_FILE_TYPES[i]):
                    self.datasets["classification"].append(stFile.replace(SUPPORTED_FILE_TYPES[i],""))
                    break
        # for stFile in os.listdir(self.datapath+"regression"):
        #     if stFile.endswith(".data"):
        #         self.datasets["regression"].append(stFile.replace(".data",""))
        #self.datasets["classification"].append("synthetic_disjunctions")
    
    def getAllDatasetNames(self):
        data = []
        for l in self.datasets.values():
            data += l
        return data
    
    def getTaskNames(self):
        return self.datasets.keys()
    
    def getDatasetTask(self, name):
        if "synthetic" in name:
                name = name[:name.find("(")]
                return name
        for task in self.datasets:            
            if name in self.datasets[task]:
                return task
        return "unknown"
    
    def getDatasetNames(self, task):
        return self.datasets[task]
    
    def getDataset(self, name, missingVal = 0.0):
        if "synthetic" in name:
            return self.createSynthetic(name)
        dataset = ""
        for root, _, files in os.walk(self.datapath):
            for filename in files:
                if name == os.path.splitext(filename)[0]:
                    dataset = os.path.join(root, filename)
        if ".arff" in dataset:
            return self.loadArff(name, dataset, missingVal)
        if ".data" in dataset:
            return self.loadSvmLightStyle(name, dataset, missingVal)
        if ".csv" in dataset:
            return self.loadCSV(name, dataset, missingVal)
        
    
    def loadArff(self, name, filename, missingVal):
        data = arff.load(open(filename))
        #attribute_names = np.array([str(x[0]) for x in data['attributes']])
        considered = []
        numerical_data = []
        labelCol = -1
        labels = []
        for i,t in enumerate(data['attributes']):
            if 'class' in str(t[0]).lower():
                labelCol = i
            else:
                att = str(t[1]).lower()
                if att in ['real', 'integer', '[u\'-1\', u\'1\', u\'0\']', '[u\'-1\', u\'1\']']:
                    considered.append(i)
        if labelCol < 0:
            print "Error: no label found. " + name
        for i, line in enumerate(data['data']):
            row = [float(line[j]) if line[j] is not None else missingVal for j in considered]
            numerical_data.append(row)
            labels.append(line[labelCol])
        classes = []
        for label in labels:
            if label not in classes:
                classes.append(label)    
        y_list = []
        for l in labels:
            y_list.append(float(classes.index(l)))
        self.datasetInfo[name] = {"N":len(numerical_data), "D":len(numerical_data[0]), "task":self.getDatasetTask(name)}
        return np.array(numerical_data), np.array(y_list)
    
    def loadCSV(self, name, filename, missingVal):
        f = open(filename, "r")        
        features = []
        labels = []
        classes = []
        count = 0
        for line in f:
            count += 1
            if count >= 100000:
                break
            vals = line.replace("\n","").split(",")
            if float(vals[0]) not in classes:
                classes.append(float(vals[0]))            
            labels.append(float(vals[0]))
            features.append([])            
            for val in vals[1:]:
                if val == None:
                    val = missingVal                
                features[-1].append(float(val))
        y_list = []
        for l in labels:
            y_list.append(float(classes.index(l)))
        featCount = len(features[0])
        instCount = len(features)
        X = np.zeros((instCount,featCount))
        for i in xrange(len(features)):
            f = features[i]            
            for j in xrange(len(f)):
                featureValue = f[j]
                X[i][j] = featureValue
        y = np.array(y_list)           
        self.datasetInfo[name] = {"N":instCount, "D":featCount, "task":self.getDatasetTask(name)}     
        return X,y
    
    def loadSvmLightStyle(self, name, filename, missingVal):        
        f = open(filename, "r")        
        features = []
        labels = []
        classes = []
        for line in f.readlines():
            vals = line.replace("\n","").split(" ")
            if vals[0] not in classes:
                classes.append(vals[0])            
            labels.append(vals[0])
            features.append(cl.defaultdict(float))            
            for val in vals[1:]:
                if ":" in val:
                    feat = val.split(":")
                    val = feat[1]
                    if val == None:
                        val = missingVal                
                    features[-1][int(feat[0])] = float(val)
        y_list = []
        for l in labels:
            y_list.append(float(classes.index(l)))
        fidx_min = 10000000000
        fidx_max = 0
        for f in features:
            feature_indizes = sorted(f.keys(), key=float)
            if float(feature_indizes[0]) < fidx_min:
                fidx_min = int(feature_indizes[0])
            if float(feature_indizes[-1]) > fidx_max:
                fidx_max = int(feature_indizes[-1])
        featCount = fidx_max - fidx_min
        instCount = len(features)
        X = np.zeros((instCount,featCount))
        for i in xrange(len(features)):
            f = features[i]            
            for j in xrange(fidx_min,fidx_max):
                featureValue = f[j]
                X[i][j-int(fidx_min)] = featureValue
        y = np.array(y_list)           
        self.datasetInfo[name] = {"N":instCount, "D":featCount, "task":self.getDatasetTask(name)}     
        return X,y
    
    def createSynthetic(self, name): #convention is to have name as "<synthetic_type>(<instCount>,<featCount>)"
        className = ""
        if "synthetic_disjunctions" in name:
            className = RapidlyDriftingDisjunction
        if "synthetic_bshouty" in name:
            className = BshoutyLongModel
        if "(" not in name:
            print "Error: " + name
            return
        args = name[name.find("(")+1:name.find(")")]
        args = args.split(",")
        instCount = int(args[0])
        dim = int(args[1])
        input_stream = className(dim, 0.0)
        X = []
        y = []
        for _ in xrange(instCount):
            (features, label) = input_stream.generate_example()
            if label == -1:
                label = 0
            X.append(features.toList(dim))
            y.append(label)
        self.datasetInfo[name] = {"N":instCount, "D":dim, "task":self.getDatasetTask(name)}
        return np.array(X),np.array(y)
    
    def getDatasetInfo(self, dataset):
        if dataset not in self.datasetInfo:
            self.getDataset(dataset)
        return self.datasetInfo[dataset]

if __name__ == "__main__":
    data = Datasets()
    print data.getAllDatasetNames()
    print data.getTaskNames()
    print data.getDatasetNames("classification")
    X,y = data.getDataset("colic")
    print X
    print y