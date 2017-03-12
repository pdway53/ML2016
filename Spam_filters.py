# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:50:51 2016

@author: way
"""

import pandas as pd
import numpy as np
import csv
import random

def run_data():
    data = pd.read_csv("spam_train.csv", header = 0)
    label = data.ix[:,58]
    train = data.drop(data.columns[[0,58]],axis = 1)
    train = noramlization(train.astype(float))
           
        
    test_data = pd.read_csv("spam_test.csv", header = None)
    x_test = test_data.drop(test_data.columns[0],axis = 1)    
    x_test = noramlization(x_test.astype(float))

    
    
    return train , label , x_test 
    

def noramlization(df):
    for col in df.columns:
        max_t = df[col].max()
        min_t = df[col].min()
        mean = df[col].mean()        
        df[col] = (df[col] - mean)/ ( max_t - min_t )

    
    num_data  = df.shape[0]
    one = [1] * num_data
    df["ones"] = one
    
    return df
        

def sigmoid(y_hat):
    y_t = 1/(1+ np.exp(-y_hat.astype(float)))
    y_t = np.round(y_t,10)
    
    return y_t
    
    

def Log_Reg(train, label , lr ,lamda ,epoch ):
    
    numTrain = 3000
    dim = train.shape[1]
    W = np.random.rand(dim ,1) 

    index_t = []
    for num in range(numTrain):
        index_t.append(num)
   
    y_train = label
    x_train = train
    
    rangeSplit = list(train.index)    
    random.shuffle(rangeSplit)
    print(rangeSplit) 
    for i in range(epoch):
      
        for row in rangeSplit:
        
            x = x_train.loc[row]
            yi = x.dot(W) 
        
            y_i = sigmoid(yi)
        
           # update w
            diff = -(y_train.loc[row] - y_i)        
            w_grad = x.apply(lambda x : x*lr*diff) # + (lamda*W/x_train.shape[0] )  regulization      
            W = W - w_grad.tolist() 
      
        y_hat = x_train.dot(W)
        yi_hat = sigmoid(y_hat)
      
        loss = -(y_train.T.dot(np.log(yi_hat)) + (1-y_train).T.dot(np.log(1-yi_hat)))
        loss = loss.astype(float)
  
        print('\epoch : %d ; Train Loss : %.5f'  %(i,loss) )
        print('\W : ' % W)
    
    return W
    
   
def predict(x_test, W):
    y = x_test.dot(W)
    y_predict = sigmoid(y)
    y_predict[y_predict >= 0.5 ] = 1
    y_predict[y_predict < 0.5 ] = 0 
    

    return y_predict

def predict_val(x_test,y_test ,W):
    y = x_test.dot(W)
    y_predict = sigmoid(y)
    y_predict[y_predict >= 0.5 ] = 1
    y_predict[y_predict < 0.5 ] = 0     
   
    acc = [ y_test.loc[i] == y_predict.loc[i].values.astype(int) for i in list(y_test.index) ]    
    accuracy =  sum(acc)/len(y_test)    
    print('acc :  %.5f '  % accuracy.astype(float))
    

    return accuracy

    
    
def train_valid_split(train, label,nfold, parmas,epoch ):    
    index = list(range(train.shape[0]))
    random.shuffle(index) 
    
    
    num = train.shape[0]
    n_each = round(num / nfold)
    vfold = []
    for i in range(nfold):
        fold = index[i*n_each:(i+1)*n_each]
        vfold.append(fold)    
           
    index_val = vfold
    
    acc_param = []
    for param in parmas:
        print('parammeter %4f' % param)
        accuracy = []  
        for ind in index_val:       
            index_num = index_val.index(ind)
            print('parammeter %4f , fold %d' % (param,index_num) )
            
            val_x = train.loc[ind]
            val_y = label.loc[ind]                
            train_x = train.drop(train.index[[ind]], inplace = False)
            train_y = label.drop(train.index[[ind]], inplace = False)                
            W = Log_Reg(train_x, train_y , 0.01,param, epoch)       
            acc = predict_val(val_x,val_y, W)      
            accuracy.append(acc) 
        acc_param.append(np.mean(accuracy))    

    print(acc_param)
    hacc_index = acc_param.index(np.max(acc_param))
    bst_param = parmas[hacc_index]
           
    return bst_param
    

def outputfile(y):
    y_predict = np.array(y)

    result = [["id","label"]]
    num = len(y_predict)   
    for it in range(num):
        ans = [it+1  , str(y_predict[it][0]) ]
        result.append(ans)
    
    csvfile = open('output.csv', 'w',newline='')
    csvwriter = csv.writer(csvfile)
    for item in result:
        csvwriter.writerow(item)
    csvfile.close()



   

def run_baseline():
    #main 

    lr = 0.01
    epoch = 200
    nfold = 5
    parmas= [0.5,0.1,0.05,0.001 ]
    
    train , label , x_test = run_data()  
    params = train_valid_split(train, label,nfold,parmas,epoch)    
    W = Log_Reg(train, label , lr, params,epoch )
    test_predict = predict(x_test, W)
    outputfile(test_predict)
    
    
run_baseline()