import numpy as np
import pandas as pd


def getDataFrame(filename, n_rows):
    '''This function reads the file provided as the argument and returns a dataframe(pandas) 
    corresponding to that without any pre-processing'''

    df = pd.read_csv(filename, delimiter='\t', nrows=n_rows)
    return df

#df = getDataFrame('./Datasets/en-gb0.tsv', 50000)
#df.to_csv('./Datasets/en-gb0-short.tsv', sep='\t')

def getValidColumns(df):
    columns = df.columns
    valid_columns = []
    for col in columns:
        if col.startswith('m:', 0, len(col)) == False:
            valid_columns.append(col)
    print('No of columns removed due to m: labels=', len(columns) - len(valid_columns))
    return valid_columns

def splitTrainTest(X, Y, split_ratio):
    trainX = X[:int(split_ratio*X.shape[0])]
    trainY = Y[:int(split_ratio*Y.shape[0])]
    testX = X[int(split_ratio*X.shape[0]):-1]
    testY = Y[int(split_ratio*Y.shape[0]):-1]
    
    trainX = trainX[:,2:]
    testX = testX[:,2:]
    return trainX, trainY, testX, testY

def preProcess(df, split_ratio):
    print(df.shape)
    df['m:IntentJudgment'] = df['m:IntentJudgment'].astype('category').cat.codes    
    dfY = df.loc[:, 'm:IntentJudgment']     

    df = df.select_dtypes(include=np.number)

    print('No of Na values= ' + str(np.sum(np.sum(df.isna()))))
    df = df.fillna(0).astype('int')
    print('No. of Na values after filling= '+ str(np.sum(np.sum(df.isna()))))

    columns = getValidColumns(df)
    dfX = df[columns]
#    try:
#        print(dfX['m:IntentJudgment'])
#    except:
#        print('m:IntentJudgment not found')
#    dfX = df.loc[:, df.columns != 'm:IntentJudgment']
    print('Before dfx shape=',dfX.shape)
    print('After dfx shape=',dfX.shape)
#    dfX = dfX.select_dtypes(include='float')
    print('columns with float values=', dfX.columns)

    print('dfY shape=', dfY.shape)
#    print(dfY)
    X = dfX.values
    Y = dfY.values
    
    return splitTrainTest(X, Y, split_ratio) 

#a, b, c, d = preProcess(getDataFrame('./Datasets/en-gb0-short.tsv', 50000), 0.7)


def getData(filename, split_ratio, n_rows=70000):
    data_frame = getDataFrame(filename, 50000)
    return preProcess(data_frame, split_ratio)

