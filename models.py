import numpy as np
import readData as rd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import _pickle as cPickle
from xgboost import XGBClassifier

def trainDT(trainX, trainY, dump_bool=0):    
    clf_DT = tree.DecisionTreeClassifier()
    clf_DT.fit(trainX, trainY)

    if dump_bool:
        try:
            with open('./Trained Models/DT.pickle', 'wb') as file:
                cPickle.dump(clf_DT, file)
        except:
            print('Some error occured during dumping the DT object')
    return clf_DT 

def trainRF(trainX, trainY, dump_bool=0):
    clf_RF = RandomForestClassifier(n_estimators=100)
    clf_RF.fit(trainX, trainY)
    
    if dump_bool:
        try:
            with open('./Trained Models/RF.pickle', 'wb') as file:
                cPickle.dump(clf_RF, file)
        except:
            print('Some error eccured during dumping the RF object')
    return clf_RF

def trainSVC(trainX, trainY, dump_bool=0):
    clf_SVC = SVC()
    clf_SVC.fit(trainX, trainY)
        
    if dump_bool:
        try:
            with open('./Trained Models/SVC.pickle', 'wb') as file:
                cPickle.dump(clf_SVC, file)
        except:
            print('Some error eccured during dumping the SVC object')
    return clf_SVC

def trainKNN(trainX, trainY, k_neighbors, weight='uniform', dump_bool=0):       #weights = 'distance'
    clf_KNN = KNeighborsClassifier(k_neighbors, weights=weight)
    clf_KNN.fit(trainX, trainY)
    file_name = str(k_neighbors) + 'NN.pickle'
    if dump_bool:
        try:
            with open('./Trained Models/' + file_name, 'wb') as file:
                cPickle.dump(clf_KNN, file)
        except:
            print('Some error eccured during dumping the KNN object')
    return clf_KNN

def trainXGB(trainX, trainY, dump_bool=0):     
    clf_XGB = XGBClassifier()
    clf_XGB.fit(trainX, trainY)
    file_name = 'XGB.pickle'
    if dump_bool:
        try:
            with open('./Trained Models/' + file_name, 'wb') as file:
                cPickle.dump(clf_XGB, file)
        except:
            print('Some error eccured during dumping the XGB object')
    return clf_XGB
