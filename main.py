import readData as rd
import _pickle as cPickle
import numpy as np
import models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

trainX, trainY, testX, testY = rd.getData('./Datasets/en-gb0-short.tsv', 0.7)

def getEvaluations(y_true, y_pred):
    results = []
    
    results.append(accuracy_score(y_true, y_pred))
    results.append(precision_score(y_true, y_pred, average='macro'))
    results.append(recall_score(y_true, y_pred, average='macro'))
    results.append(f1_score(y_true, y_pred, average='macro'))
    
    return results


def getTrainedDT(dump_bool=0):
    try:
        with open('./Trained Models/DT.pickle', 'rb') as file:
            clf_DT = cPickle.load(file)
        return clf_DT
    except:
        print('Trained DT model not found, Re-training...')
        return models.trainDT(trainX, trainY, dump_bool)

def getTrainedRF(dump_bool=0):
    try:
        with open('./Trained Models/RF.pickle', 'rb') as file:
            clf_RF = cPickle.load(file)
        return clf_RF
    except:
        print('Trained RF model not found, Re-training...')
        return models.trainRF(trainX, trainY, dump_bool)


def getTrainedSVC(dump_bool=0):
    try:
        with open('./Trained Models/SVC.pickle', 'rb') as file:
            clf_SVC = cPickle.load(file)
        return clf_SVC
    except:
        print('Trained SVC model not found, Re-training...')
        return models.trainSVC(trainX, trainY, dump_bool)


def getTrainedKNN(k_neighbors, dump_bool=0, weight='uniform' ):
    file_name = str(k_neighbors) + 'NN.pickle'
    try:
        with open('./Trained Models/' + file_name, 'rb') as file:
            clf_KNN = cPickle.load(file)
        return clf_KNN
    except:
        print('Trained KNN model not found, Re-training...')
        return models.trainKNN(trainX, trainY, k_neighbors, weight, dump_bool)

def getTrainedXGB(dump_bool=0):
    file_name =  'XGB.pickle'
    try:
        with open('./Trained Models/' + file_name, 'rb') as file:
            clf_XGB = cPickle.load(file)
        return clf_XGB
    except:
        print('Trained XGB model not found, Re-training...')
        return models.trainXGB(trainX, trainY, dump_bool)





clf_DT = getTrainedDT()
clf_DT_pred = clf_DT.predict(testX)

clf_RF = getTrainedRF()
clf_RF_pred = clf_RF.predict(testX)

clf_KNN = getTrainedKNN(10, 1, 'uniform')
clf_KNN_pred = clf_KNN.predict(testX)


clf_XGB = getTrainedXGB(1)
clf_XGB_pred = clf_XGB.predict(testX)

print('DT:',getEvaluations(testY, clf_DT_pred))
print('RF:',getEvaluations(testY, clf_RF_pred))
print('KNN:',getEvaluations(testY, clf_KNN_pred))
print('XGB:',getEvaluations(testY, clf_XGB_pred))

