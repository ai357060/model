import numpy as np
import pandas as pd
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import datetime
import multiprocessing
import time
import os
from fastai.tabular import *
import glob

#import threading 
#import ctypes 
#import time 

def loaddata_master(datafile):
    df = pd.read_csv(datafile,sep=';')
    try:
        df.fulldate=pd.to_datetime(df.fulldate,format='%Y-%m-%d')
    except:
        df.fulldate=pd.to_datetime(df.fulldate,format='%Y.%m.%d')    

    df.insert(2,'id',df.index)
    return df

def asynchfit(model, X_train, y_train, resultdict): 
    model.fit(X_train, y_train)
    resultdict[0] = model
    
def timelimitfit(model,X_train, y_train,timeout = 30):    
    manager = multiprocessing.Manager()
    resultdict = manager.dict()
    process = multiprocessing.Process(target=asynchfit, args=(model,X_train,y_train,resultdict,)) 
    process.start() 
    process.join(timeout*60)
    process.terminate() 
    if len(resultdict) > 0:
        return resultdict[0]
    else:
        raise Exception('FitTimeOut: '+str(timeout)+'min')
        
def plot_feature_importances(X_test,feature_importances,feature_names):
    n_features = X_test.shape[1]
    plt.figure(figsize=(10,100))
    plt.barh(np.arange(n_features), feature_importances, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)        
        
def ExamineLogisticRegression(masterframe,Xintex,X_train, y_train,X_test, y_test,featurenames,testone,plot):
#logistics regresion (classification)
#default C=1;  lower values of C correspond to more regularization. Regularization means explicitly restricting a model to avoid overfitting.
#solver = ‘liblinear’  {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
#penalty default l2
#max_iter =100
    linear_resdf = pd.DataFrame(columns=['proctime','penalty','solver','C','max_iter','Tr acc','Te acc','Te_cal','Mess'
                                         ,'Tr_1_cnt','Te_1_cnt','Tr_0_acc','Tr_1_acc','Te_cnt'
,'Te_0_50','Te_0_50_cnt','Te_0_60','Te_0_60_cnt','Te_0_70','Te_0_70_cnt','Te_0_80','Te_0_80_cnt','Te_0_90','Te_0_90_cnt'
,'Te_1_50','Te_1_50_cnt','Te_1_60','Te_1_60_cnt','Te_1_70','Te_1_70_cnt','Te_1_80','Te_1_80_cnt','Te_1_90','Te_1_90_cnt'])

    penaltynames = ['l1','l2']
    solvers = [['liblinear','saga'],['lbfgs','newton-cg','sag']]
    CC = [0.0001,0.001,0.01,1,10,100]  #[0.001,0.01,0.1,1,10,100]
    max_iters = [100]                 #[100,1000,10000]

    if testone == True:
        penaltynames = ['l1']
        solvers = [['liblinear']]
        CC = [0.0001]  #[0.001,0.01,0.1,1,10,100]
        max_iters = [100]                 #[100,1000,10000]
    
    atr = masterframe.atr14tr.mean()    
    for penalty in range(0,len(penaltynames)):
        for solver in solvers[penalty]:
            for C in CC:                
                for max_iter in max_iters:               
                    try:
                        with warnings.catch_warnings(record=True) as efitwarn:
                            starttime = datetime.datetime.now()
                            print(starttime,penaltynames[penalty],solver,C,max_iter)
                            logreg = LogisticRegression(C=C,solver=solver,max_iter = max_iter, penalty=penaltynames[penalty]
                                                       ,n_jobs=1)
                            logreg.fit(X_train, y_train)
                            #logreg = timelimitfit(logreg,X_train, y_train,30)
                            endtime = datetime.datetime.now()
                            if len(efitwarn) > 0: 
                                fitwarn = efitwarn[-1].message 
                            else: 
                                fitwarn = ''
                    except Exception as e:
                        print('Exception: ',e)
                        endtime = datetime.datetime.now()
                        linear_resdf = linear_resdf.append({
                            'proctime':str(endtime - starttime),
                            'C':C,
                            'solver':solver,
                            'max_iter':max_iter,
                            'penalty':penaltynames[penalty],
                            'Mess':str(e)
                        },ignore_index=True)                        
                        continue

                    if plot: plot_feature_importances(X_test,logreg.feature_importances_,featurenames)

                    linear_resdf1 = {
                        'proctime':str(endtime - starttime),
                        'C':C,
                        'solver':solver,
                        'max_iter':max_iter,
                        'penalty':penaltynames[penalty]
                    }
                    linear_resdf2 = PrepareResults(masterframe,logreg,X_train,X_test,y_train,y_test,Xintex,testone,fitwarn,atr,str(endtime - starttime))
                    linear_resdf1.update(linear_resdf2)
                    linear_resdf = linear_resdf.append(linear_resdf1 ,ignore_index=True)

    return linear_resdf

def ExamineLinearSVC(masterframe,Xintex,X_train, y_train,X_test, y_test,featurenames,testone,plot):
    linear_resdf = pd.DataFrame(columns=['proctime','penalty','C','max_iter','loss','Tr acc','Te acc','Te_cal','Mess'
                                         ,'Tr_1_cnt','Te_1_cnt','Tr_0_acc','Tr_1_acc','Te_cnt'
,'Te_0_50','Te_0_50_cnt','Te_0_60','Te_0_60_cnt','Te_0_70','Te_0_70_cnt','Te_0_80','Te_0_80_cnt','Te_0_90','Te_0_90_cnt'
,'Te_1_50','Te_1_50_cnt','Te_1_60','Te_1_60_cnt','Te_1_70','Te_1_70_cnt','Te_1_80','Te_1_80_cnt','Te_1_90','Te_1_90_cnt'])

    penaltynames = ['l2']
    loss = [['hinge','squared_hinge']]
    CC = [0.0001,0.001,0.01,1,10,100]  #[0.001,0.01,0.1,1,10,100]
    max_iters = [10000]                 #[100,1000,10000]

    if testone == True:
        penaltynames = ['l2']
        loss = [['squared_hinge']]
        CC = [0.0001]  #[0.001,0.01,0.1,1,10,100]
        max_iters = [100]                 #[100,1000,10000]
    
    
    atr = masterframe.atr14tr.mean()    
    for penalty in range(0,len(penaltynames)):
        for loss in loss[penalty]:
            for C in CC:                
                for max_iter in max_iters:               
                    try:
                        with warnings.catch_warnings(record=True) as efitwarn:
                            starttime = datetime.datetime.now()
                            print(starttime,penalty,loss,C,max_iter)
                            svm = LinearSVC(C=C,max_iter = max_iter, penalty=penaltynames[penalty],loss=loss)
                            logreg = CalibratedClassifierCV(svm) 
                            logreg.fit(X_train, y_train)
                            #logreg = timelimitfit(logreg,X_train, y_train,30)
                            endtime = datetime.datetime.now()
                            if len(efitwarn) > 0: 
                                fitwarn = efitwarn[-1].message 
                            else: 
                                fitwarn = ''
                    except Exception as e:
                        print('Exception: ',e)
                        endtime = datetime.datetime.now()
                        linear_resdf = linear_resdf.append({
                            'proctime':str(endtime - starttime),
                            'C':C,
                            'max_iter':max_iter,
                            'penalty':penaltynames[penalty],
                            'loss':loss,
                            'Mess':str(e)
                        },ignore_index=True)                        
                        continue

                    if plot: plot_feature_importances(X_test,logreg.feature_importances_,featurenames)

                    linear_resdf1 = {
                        'proctime':str(endtime - starttime),
                        'C':C,
                        'max_iter':max_iter,
                        'penalty':penaltynames[penalty],
                        'loss':loss
                    }
                    linear_resdf2 = PrepareResults(masterframe,logreg,X_train,X_test,y_train,y_test,Xintex,testone,fitwarn,atr,str(endtime - starttime))
                    linear_resdf1.update(linear_resdf2)
                    linear_resdf = linear_resdf.append(linear_resdf1 ,ignore_index=True)
#                     print('train: {:10.1f} | test: {:10.1f} | proctime: {}'.format(logreg.score(X_train, y_train)*100
#                                                                     , logreg.score(X_test, y_test)*100
#                                                                    ,str(endtime - starttime)))

    return linear_resdf


def ExamineRandomForest(masterframe,Xintex,X_train, y_train,X_test, y_test,featurenames,testone,plot,automaxfeat=False):
    # n_estimators - the more the better
    # max_features - default=sqrt(n_features)
    # max_depth 
    # max_leaf_nodes
    
    linear_resdf = pd.DataFrame(columns=['proctime','max_features','n_estimators','max_depth','max_leaf_nodes','Tr acc','Te acc','Te_cal'
                                         ,'Mess'
                                         ,'Tr_1_cnt','Te_1_cnt','Tr_0_acc','Tr_1_acc','Te_cnt'
,'Te_0_50','Te_0_50_cnt','Te_0_60','Te_0_60_cnt','Te_0_70','Te_0_70_cnt','Te_0_80','Te_0_80_cnt','Te_0_90','Te_0_90_cnt'
,'Te_1_50','Te_1_50_cnt','Te_1_60','Te_1_60_cnt','Te_1_70','Te_1_70_cnt','Te_1_80','Te_1_80_cnt','Te_1_90','Te_1_90_cnt'])
    
#     max_features = [5,20,60,100,150,300,400]  #'auto'
    max_features = ['auto','log2',None]  #'auto'
    n_estimators = [100,400,700,900]
    max_depth = [5,30,70,100,300]   #None
    max_leaf_nodes = [5,10,50,100,300] #None

    if testone == True:
        max_features = [None]  
        n_estimators = [100]
        max_depth = [30]   
        max_leaf_nodes = [10] 

    if (automaxfeat == True):
        max_features = [None]

    atr = masterframe.atr14tr.mean()    
    for i_max_features in max_features:
        for i_n_estimators in n_estimators:
            for i_max_depth in max_depth:                
                for i_max_leaf_nodes in max_leaf_nodes:              
                    try:
                        with warnings.catch_warnings(record=True) as efitwarn:
                            starttime = datetime.datetime.now()
                            print(starttime,i_max_features,i_n_estimators,i_max_depth,i_max_leaf_nodes)
                            logreg = RandomForestClassifier(max_features=i_max_features, n_estimators=i_n_estimators
                                                            ,max_depth=i_max_depth, max_leaf_nodes=i_max_leaf_nodes,random_state=2
                                                            ,n_jobs=1)
                            logreg.fit(X_train, y_train)
                            #logreg = timelimitfit(logreg,X_train, y_train,30)
                            endtime = datetime.datetime.now()
                            if len(efitwarn) > 0: 
                                fitwarn = efitwarn[-1].message 
                            else: 
                                fitwarn = ''
                    except Exception as e:
                        print('Exception: ',e)
                        endtime = datetime.datetime.now()
                        linear_resdf = linear_resdf.append({
                            'proctime':str(endtime - starttime),
                            'max_features':i_max_features,
                            'n_estimators':i_n_estimators,
                            'max_depth':i_max_depth,
                            'max_leaf_nodes':i_max_leaf_nodes,
                            'Mess':str(e)
                        },ignore_index=True)                        
                        continue

                    if plot: plot_feature_importances(X_test,logreg.feature_importances_,featurenames)

                    linear_resdf1 = {
                        'proctime':str(endtime - starttime),
                        'max_features':i_max_features,
                        'n_estimators':i_n_estimators,
                        'max_depth':i_max_depth,
                        'max_leaf_nodes':i_max_leaf_nodes
                    }
                    linear_resdf2 = PrepareResults(masterframe,logreg,X_train,X_test,y_train,y_test,Xintex,testone,fitwarn,atr,str(endtime - starttime))
                    linear_resdf1.update(linear_resdf2)
                    linear_resdf = linear_resdf.append(linear_resdf1 ,ignore_index=True)
#                     print('train: {:10.1f} | test: {:10.1f} | proctime: {}'.format(logreg.score(X_train, y_train)*100
#                                                                     , logreg.score(X_test, y_test)*100
#                                                                    ,str(endtime - starttime)))
                   
    return linear_resdf



def ExamineSVC(masterframe,Xintex,X_train, y_train,X_test, y_test,featurenames,testone,plot):
    # n_estimators - the more the better
    # max_features - default=sqrt(n_features)
    # max_depth 
    # max_leaf_nodes
    
    linear_resdf = pd.DataFrame(columns=['proctime','max_iter','C','gamma','kernel','degree','Tr acc','Te acc','Te_cal','Mess'
                                         ,'Tr_1_cnt','Te_1_cnt','Tr_0_acc','Tr_1_acc','Te_cnt'
,'Te_0_50','Te_0_50_cnt','Te_0_60','Te_0_60_cnt','Te_0_70','Te_0_70_cnt','Te_0_80','Te_0_80_cnt','Te_0_90','Te_0_90_cnt'
,'Te_1_50','Te_1_50_cnt','Te_1_60','Te_1_60_cnt','Te_1_70','Te_1_70_cnt','Te_1_80','Te_1_80_cnt','Te_1_90','Te_1_90_cnt'])
    

    kernel = ['rbf','linear', 'poly',  'sigmoid'] #None # dla poly trzeba chyba inne parametry. Przy C = 100 i gamma = 1 robi się bardzo długo
    C = [0.001,0.01,0.1,1,10,100,1000]
    gamma = ['scale','auto',0.0001,0.001,0.01,0.1,1,10]   #None gamma = 1 długo przy degree 5, 
    max_iter = [-1]  #'auto'
    degree = [2,3,4,5] # degree=5 wolno dla poly gdy C>=100
    timeout = 5
    
    if testone == True:
        kernel = ['rbf'] 
        C = [100]
        gamma = [0.01]   
        max_iter = [-1]  
        degree = [2]
        timeout = 5
    
    atr = masterframe.atr14tr.mean()    
    for i_kernel in kernel:
        for i_C in C:
            for i_gamma in gamma:                
                for i_max_iter in max_iter:   
                    for i_degree in degree:
                        if ((i_kernel != 'poly') & (i_degree > 2)):
                            continue
                        if ((i_kernel == 'poly') & (i_degree > 2)& (i_gamma == 10)& (i_C >= 1.0)):
                            continue
                
                        try:
                            with warnings.catch_warnings(record=True) as efitwarn:
                                starttime = datetime.datetime.now()
                                print(starttime,i_kernel,i_C,i_gamma,i_max_iter,i_degree)
                                logreg = SVC(degree = i_degree, max_iter=i_max_iter, C=i_C,gamma=i_gamma
                                             , kernel=i_kernel,probability=True,random_state=2)
                                if ((i_kernel == 'poly')|(i_kernel == 'linear')):
                                    logreg = timelimitfit(logreg,X_train, y_train,timeout)
                                else:
                                    logreg.fit(X_train, y_train)
                                endtime = datetime.datetime.now()
                                if len(efitwarn) > 0: 
                                    fitwarn = efitwarn[-1].message 
                                else: 
                                    fitwarn = ''
                        except Exception as e:
                            print('Exception: ',e)
                            endtime = datetime.datetime.now()
                            linear_resdf = linear_resdf.append({
                                'proctime':str(endtime - starttime),
                                'max_iter':i_max_iter,
                                'C':i_C,
                                'gamma':i_gamma,
                                'kernel':i_kernel,
                                'degree':i_degree,
                                'Mess':str(e)
                            },ignore_index=True)                        
                            continue

                        if plot: plot_feature_importances(X_test,logreg.feature_importances_,featurenames)
                            
                        linear_resdf1 = {
                            'proctime':str(endtime - starttime),
                            'max_iter':i_max_iter,
                            'C':i_C,
                            'gamma':i_gamma,
                            'kernel':i_kernel,
                            'degree':i_degree
                        }
                        linear_resdf2 = PrepareResults(masterframe,logreg,X_train,X_test,y_train,y_test,Xintex,testone,fitwarn,atr,str(endtime - starttime))
                        linear_resdf1.update(linear_resdf2)
                        linear_resdf = linear_resdf.append(linear_resdf1 ,ignore_index=True)
#                         print('train: {:10.1f} | test: {:10.1f} | proctime: {}'.format(logreg.score(X_train, y_train)*100
#                                                                         , logreg.score(X_test, y_test)*100
#                                                                        ,str(endtime - starttime)))
                   
    return linear_resdf

def ExamineMLP(masterframe,Xintex,X_train, y_train,X_test, y_test,featurenames,testone,plot):
    # n_estimators - the more the better
    # max_features - default=sqrt(n_features)
    # max_depth 
    # max_leaf_nodes
    
    linear_resdf = pd.DataFrame(columns=['proctime','solver','layers','activation','max_iter','alpha','Tr acc','Te acc','Te_cal','Mess'
                                         ,'Tr_1_cnt','Te_1_cnt','Tr_0_acc','Tr_1_acc','Te_cnt'
,'Te_0_50','Te_0_50_cnt','Te_0_60','Te_0_60_cnt','Te_0_70','Te_0_70_cnt','Te_0_80','Te_0_80_cnt','Te_0_90','Te_0_90_cnt'
,'Te_1_50','Te_1_50_cnt','Te_1_60','Te_1_60_cnt','Te_1_70','Te_1_70_cnt','Te_1_80','Te_1_80_cnt','Te_1_90','Te_1_90_cnt'])
    
    solver = ['lbfgs']#['adam','lbfgs','sgd'] #None
#     hidden_layer_sizes = [[10],[100],[400],[800]
#                           ,[10,10],[100,100],[400,400],[800,800]
#                           ,[10,10,10],[100,100,100],[400,400,400],[800,800,800]
#                           ,[10,10,10,10],[100,100,100,100],[400,400,400,400],[800,800,800,800]]
    hidden_layer_sizes = [[10],[100],[400],[800]
                          ,[10,10],[100,100],[400,400]
                          ,[10,10,10],[100,100,100]
                          ,[10,10,10,10],[100,100,100,100]]    
    activation = ['tanh', 'relu']   # ['tanh','identity', 'logistic', 'relu']
    alpha = [0.00001,0.0001,0.01,0.1,1]  #'auto'
    max_iter= [1000] # [100,1000]
# learning_rate_init : double, optional, default 0.001 The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.
# shuffle : bool, optional, default True
# beta_1 : float, optional, default 0.9
# Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver=’adam’

# beta_2 : float, optional, default 0.999
# Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver=’adam’

# epsilon : float, optional, default 1e-8
# Value for numerical stability in adam. Only used when solver=’adam’

# test one model
    if testone == True:
        solver = ['lbfgs']#,'lbfgs']#['adam','lbfgs','sgd'] #None
        hidden_layer_sizes = [[10]]
        activation = ['logistic']#,'identity', 'logistic', 'tanh', 'relu']   # ['tanh','identity', 'logistic', 'tanh', 'relu']
        alpha = [0.01]#,0.01,0.1,1]  #'auto'
        max_iter= [1000]    
    
    atr = masterframe.atr14tr.mean()    
    for i_solver in solver:
        for i_hidden_layer_sizes in hidden_layer_sizes:
            for i_activation in activation:                
                for i_alpha in alpha:  
                    for i_max_iter in max_iter:
                        try:
                            with warnings.catch_warnings(record=True) as efitwarn:
                                starttime = datetime.datetime.now()
                                print(starttime,i_solver,i_hidden_layer_sizes,i_activation,i_alpha,i_max_iter)
                                logreg = MLPClassifier(solver=i_solver,hidden_layer_sizes=i_hidden_layer_sizes
                                                       ,activation=i_activation, alpha=i_alpha
                                                       ,max_iter = i_max_iter, random_state=0)
                                # logreg.fit(X_train, y_train)
                                logreg = timelimitfit(logreg,X_train, y_train,5)
                                endtime = datetime.datetime.now()
                                if len(efitwarn) > 0: 
                                    fitwarn = efitwarn[-1].message
                                    print(fitwarn)
                                else: 
                                    fitwarn = ''
                        except Exception as e:
                            print('Exception: ',e)
                            endtime = datetime.datetime.now()
                            linear_resdf = linear_resdf.append({
                                'proctime':str(endtime - starttime),
                                'solver':i_solver,
                                'layers':i_hidden_layer_sizes,
                                'activation':i_activation,
                                'alpha':i_alpha,
                                'max_iter':i_max_iter,
                                'Mess':str(e)
                            },ignore_index=True)                        
                            continue
                            
                        if plot: plot_feature_importances(X_test,logreg.feature_importances_,featurenames)
                            
                        linear_resdf1 = {
                            'proctime':str(endtime - starttime),
                            'solver':i_solver,
                            'layers':i_hidden_layer_sizes,
                            'activation':i_activation,
                            'alpha':i_alpha,
                            'max_iter':i_max_iter
                        }
                        linear_resdf2 = PrepareResults(masterframe,logreg,X_train,X_test,y_train,y_test,Xintex,testone,fitwarn,atr,str(endtime - starttime))
                        linear_resdf1.update(linear_resdf2)
                        linear_resdf = linear_resdf.append(linear_resdf1 ,ignore_index=True)
#                         print('train: {:10.1f} | test: {:10.1f} | proctime: {}'.format(logreg.score(X_train, y_train)*100
#                                                                         , logreg.score(X_test, y_test)*100
#                                                                        ,str(endtime - starttime)))
    return linear_resdf

def ExamineNN(masterframe,Xintex,datamasterframe,featurenames,testone,plot):

    cwd = os.getcwd()
    path = cwd
    dep_var = datamasterframe.columns[-1]
    cat_names = []
    cont_names = datamasterframe.columns[1:-1]
#     dep_var = 'y'
#     cat_names = ['month__1','month__2','month__3','month__4','month__5','month__6','month__7','month__8','month__9','month__10','month__11','month__12','day__1','day__2','day__3','day__4','day__5','day__6','day__7','day__8','day__9','day__10','day__11','day__12','day__13','day__14','day__15','day__16','day__17','day__18','day__19','day__20','day__21','day__22','day__23','day__24','day__25','day__26','day__27','day__28','day__29','day__30','day__31','weekday__1','weekday__2','weekday__3','weekday__4','weekday__6','weekday__7']
#     cont_names = set(featurenames) - set(cat_names)
    procs = [FillMissing, Categorify, Normalize]

    linear_resdf = pd.DataFrame(columns=['proctime','emb_drop','layers','lr','wd','epoch','Tr acc','Te acc','Te_cal','Mess'
                                         ,'Tr_1_cnt','Te_1_cnt','Tr_0_acc','Tr_1_acc','Te_cnt'
,'Te_0_50','Te_0_50_cnt','Te_0_60','Te_0_60_cnt','Te_0_70','Te_0_70_cnt','Te_0_80','Te_0_80_cnt','Te_0_90','Te_0_90_cnt'
,'Te_1_50','Te_1_50_cnt','Te_1_60','Te_1_60_cnt','Te_1_70','Te_1_70_cnt','Te_1_80','Te_1_80_cnt','Te_1_90','Te_1_90_cnt'])
    
    data = (TabularList.from_df(datamasterframe, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
#                            .split_by_rand_pct(valid_pct=0.25, seed=42)
#                            .split_by_idx(list(range(3000,4115)))
                           .split_by_idx(datamasterframe[datamasterframe[0].isin(Xintex.astype(int))].index)
                           .label_from_df(cols=dep_var)
                           .databunch(bs=64))
    
    emb_drop = [0.2,0.5,0.7]
    hidden_layer_sizes = [[10],[100],[400],[800]
                          ,[10,10],[100,100],[400,400],[800,800]
                          ,[10,10,10],[100,100,100],[400,400,400],[800,800,800]
                          ,[10,10,10,10],[100,100,100,100],[400,400,400,400],[800,800,800,800]]
    lr = [1e-4,1e-3,1e-2]
    wd = [0.5,0.1,0.01]
    epoch = [3,6,9]
    
#     learn = tabular_learner(data, layers=[400,200,100], metrics=accuracy, emb_drop=0.7,wd=0.01, silent = False)
#     learn.lr_find(end_lr=100)
#     learn.recorder.plot()
    

    
    atr = masterframe.atr14tr.mean()    
    for i_emb_drop in emb_drop:
        for i_hidden_layer_sizes in hidden_layer_sizes:
            for i_lr in lr:                
                for i_wd in wd:  
                    for i_epoch in epoch:
                        try:
                            with warnings.catch_warnings(record=True) as efitwarn:
                                starttime = datetime.datetime.now()
                                print(starttime,i_emb_drop,i_hidden_layer_sizes,i_lr,i_wd,i_epoch)
#                                 logreg = MLPClassifier(solver=i_solver,hidden_layer_sizes=[10,10]
#                                                        ,activation=i_activation, alpha=i_alpha
#                                                        ,max_iter = i_max_iter, random_state=0)
#                                 logreg.fit(X_train, y_train)
                                logreg = tabular_learner(data, emb_drop=i_emb_drop, layers=i_hidden_layer_sizes, wd=i_wd, metrics=accuracy, silent = False)
                                logreg.fit(i_epoch,lr=i_lr)
                                endtime = datetime.datetime.now()
                                if len(efitwarn) > 0: 
                                    fitwarn = efitwarn[-1].message
                                    print(fitwarn)
                                else: 
                                    fitwarn = ''
                        except Exception as e:
                            print('Exception: ',e)
                            endtime = datetime.datetime.now()
                            linear_resdf = linear_resdf.append({
                                'proctime':str(endtime - starttime),
                                'solver':i_solver,
                                'layers':i_hidden_layer_sizes,
                                'activation':i_activation,
                                'alpha':i_alpha,
                                'max_iter':i_max_iter,
                                'Mess':str(e)
                            },ignore_index=True)                        
                            continue
                            
                        if plot: plot_feature_importances(X_test,logreg.feature_importances_,featurenames)
                            
                        linear_resdf1 = {
                            'proctime':str(endtime - starttime),
                            'solver':i_solver,
                            'layers':i_hidden_layer_sizes,
                            'activation':i_activation,
                            'alpha':i_alpha,
                            'max_iter':i_max_iter
                        }
                        linear_resdf2 = PrepareResults(masterframe,logreg,X_train,X_test,y_train,y_test,Xintex,testone,fitwarn,atr,str(endtime - starttime))
                        linear_resdf1.update(linear_resdf2)
                        linear_resdf = linear_resdf.append(linear_resdf1 ,ignore_index=True)
    
    
    return linear_resdf


def PrepareResults(masterframe,logreg,X_train,X_test,y_train,y_test,Xintex,testone,fitwarn,atr,proctime):
    y_train_predict = logreg.predict(X_train)
    y_test_predict = logreg.predict(X_test)
    test_size = X_test.shape[0]
    proba = logreg.predict_proba(X_test)
    proba0 = proba[:,0]
    proba1 = proba[:,1]
    zerocnt50 = np.sum( proba0 > 0.5) 
    zerohit50 = np.sum((y_test == 0) & (proba0 > 0.5))
    zerocnt60 = np.sum( proba0 > 0.6) 
    zerohit60 = np.sum((y_test == 0) & (proba0 > 0.6))
    zerocnt70 = np.sum( proba0 > 0.7) 
    zerohit70 = np.sum((y_test == 0) & (proba0 > 0.7))
    zerocnt80 = np.sum( proba0 > 0.8) 
    zerohit80 = np.sum((y_test == 0) & (proba0 > 0.8))
    zerocnt90 = np.sum( proba0 > 0.9) 
    zerohit90 = np.sum((y_test == 0) & (proba0 > 0.9))

    onecnt50 = np.sum( proba1 >= 0.5) 
    onehit50 = np.sum((y_test == 1) & (proba1 >= 0.5))
    onecnt60 = np.sum( proba1 >= 0.6) 
    onehit60 = np.sum((y_test == 1) & (proba1 >= 0.6))
    onecnt70 = np.sum( proba1 >= 0.7) 
    onehit70 = np.sum((y_test == 1) & (proba1 >= 0.7))
    onecnt80 = np.sum( proba1 >= 0.8) 
    onehit80 = np.sum((y_test == 1) & (proba1 >= 0.8))
    onecnt90 = np.sum( proba1 >= 0.9) 
    onehit90 = np.sum((y_test == 1) & (proba1 >= 0.9))

    tecal = (zerocnt70+onecnt70)/len(y_test)*100
    res1, profitdf1 = ExamineProfit1(masterframe, Xintex, y_test, proba0,proba1,0,tecal,atr)
    res2, profitdf2 = ExamineProfit2(masterframe, Xintex, y_test, proba0,proba1,0,tecal,atr)
    res3, profitdf3 = ExamineProfit3(masterframe, Xintex, y_test, proba0,proba1,0,tecal,atr)
    res4, profitdf4 = ExamineProfit4(masterframe, Xintex, y_test, proba0,proba1,0,tecal,atr)
    if testone == True:
        profitdf1.to_csv(sep=';',path_or_buf='../Resu/temp_EP1_'+str(int(time.time()))+'.csv',date_format="%Y-%m-%d")
        profitdf2.to_csv(sep=';',path_or_buf='../Resu/temp_EP2_'+str(int(time.time()))+'.csv',date_format="%Y-%m-%d")
        profitdf3.to_csv(sep=';',path_or_buf='../Resu/temp_EP3_'+str(int(time.time()))+'.csv',date_format="%Y-%m-%d")
        profitdf4.to_csv(sep=';',path_or_buf='../Resu/temp_EP4_'+str(int(time.time()))+'.csv',date_format="%Y-%m-%d")

    linear_resdict = {
        'Tr acc': "{:10.1f}".format(logreg.score(X_train, y_train)*100),
        'Te acc': "{:10.1f}".format(logreg.score(X_test, y_test)*100),
        'Te_cal': "{:10.1f}".format(tecal),
        'Mess': fitwarn,
        'Tr_1_cnt': "{:10.1f}".format(len(y_train[y_train==1])/len(y_train)*100),
        'Tr_0_acc': "{:10.1f}".format((y_train_predict[y_train==0]).tolist().count(0)/len(y_train[y_train==0])*100),
        'Tr_1_acc': "{:10.1f}".format((y_train_predict[y_train==1]).tolist().count(1)/len(y_train[y_train==1])*100),
        'Te_1_cnt': "{:10.1f}".format((y_test == 1).tolist().count(True)/len(y_test)*100),
        'Te_cnt': "{:10.0f}".format(test_size),
        'Te_0_50' : "{:10.1f}".format(zerohit50/zerocnt50*100 if zerocnt50>0 else 0),
        'Te_0_50_cnt' : "{:10.0f}".format(zerocnt50/test_size*100),
        'Te_0_60' : "{:10.1f}".format(zerohit60/zerocnt60*100 if zerocnt60>0 else 0),
        'Te_0_60_cnt' : "{:10.0f}".format(zerocnt60/test_size*100),
        'Te_0_70' : "{:10.1f}".format(zerohit70/zerocnt70*100 if zerocnt70>0 else 0),
        'Te_0_70_cnt' : "{:10.0f}".format(zerocnt70/test_size*100),
        'Te_0_80' : "{:10.1f}".format(zerohit80/zerocnt80*100 if zerocnt80>0 else 0),
        'Te_0_80_cnt' : "{:10.0f}".format(zerocnt80/test_size*100),
        'Te_0_90' : "{:10.1f}".format(zerohit90/zerocnt90*100 if zerocnt90>0 else 0),
        'Te_0_90_cnt' : "{:10.0f}".format(zerocnt90/test_size*100),
        'Te_1_50' : "{:10.1f}".format(onehit50/onecnt50*100 if onecnt50>0 else 0),
        'Te_1_50_cnt' : "{:10.0f}".format(onecnt50/test_size*100),
        'Te_1_60' : "{:10.1f}".format(onehit60/onecnt60*100 if onecnt60>0 else 0),
        'Te_1_60_cnt' : "{:10.0f}".format(onecnt60/test_size*100),
        'Te_1_70' : "{:10.1f}".format(onehit70/onecnt70*100 if onecnt70>0 else 0),
        'Te_1_70_cnt' : "{:10.0f}".format(onecnt70/test_size*100),
        'Te_1_80' : "{:10.1f}".format(onehit80/onecnt80*100 if onecnt80>0 else 0),
        'Te_1_80_cnt' : "{:10.0f}".format(onecnt80/test_size*100),
        'Te_1_90' : "{:10.1f}".format(onehit90/onecnt90*100 if onecnt90>0 else 0),
        'Te_1_90_cnt' : "{:10.0f}".format(onecnt90/test_size*100)
#         'Str1_0.5':res1[0.5],'Str1_0.6':res1[0.6],'Str1_0.7':res1[0.7],'Str1_0.8':res1[0.8],'Str1_0.9':res1[0.9],
#         'Str2_0.5':res2[0.5],'Str2_0.6':res2[0.6],'Str2_0.7':res2[0.7],'Str2_0.8':res2[0.8],'Str2_0.9':res2[0.9],
#         'Str3_0.5':res3[0.5],'Str3_0.6':res3[0.6],'Str3_0.7':res3[0.7],'Str3_0.8':res3[0.8],'Str3_0.9':res3[0.9],
#         'Str4_0.5':res4[0.5],'Str4_0.6':res4[0.6],'Str4_0.7':res4[0.7],'Str4_0.8':res4[0.8],'Str4_0.9':res4[0.9]
    }
    linear_resdict.update(res1)
    linear_resdict.update(res2)
    linear_resdict.update(res3)
    linear_resdict.update(res4)
    
    print('train: {:10.1f} | test: {:10.1f} | cal: {:10.1f} | proctime: {}'.format(logreg.score(X_train, y_train)*100
                                                    , logreg.score(X_test, y_test)*100
                                                    , tecal
                                                   ,proctime))
    
#     for f in glob.glob("spin*"):
#         os.remove(f)
#     filename = 'spin'+ datetime.datetime.now().strftime("%d%m%y %H%M%S")
#     open(filename,'x').close()
    
    return linear_resdict

def ExamineProfit1(masterframe, Xintex, y_test, proba0, proba1, trade_cnt_th ,tecal,atr):
# SELL
# a. predict = 0 - trade
#     predict = 1 - NO trade
#     sl = high - close
#     tp  = close - nextclose

#     atr = masterframe.atr14tr.mean()
    minslratio = 0.05 
    strname = 'S1_'
    profitdf = pd.DataFrame({'id':(Xintex).astype(int),'y_test':y_test,'proba0':proba0,'proba1':proba1})
    profitdf = profitdf.set_index(profitdf.id)
    profitdf = profitdf.drop(['id'],1)
    profitdf['close'] = masterframe.close
    profitdf['nextclose'] = masterframe.close.shift(-1)
    profitdf['nexthigh'] = masterframe.high.shift(-1)
    profitdf['high'] = masterframe.high
    profitdf['sl'] = masterframe.high - masterframe.close
    profitdf['tp'] = profitdf.close - profitdf.nextclose
    profitdf['tp'] = np.where(profitdf.tp>atr,atr,profitdf.tp)
    profitdf['result'] = np.where(profitdf.nexthigh >= profitdf.high,-1 * profitdf.sl,profitdf.tp)
    res = {}
    for treshold in [0.5,0.6,0.7,0.8,0.9]:
        profitdf['result'+str(treshold)] = profitdf.result.iloc[np.where((profitdf.proba0 >= treshold)
                                                                            &(profitdf.sl>=atr*minslratio),True,False)]
        profitdf['result'+str(treshold)] = profitdf['result'+str(treshold)] / profitdf.sl
        loss = np.sum(profitdf['result'+str(treshold)]==-1)
        overall  = np.sum(profitdf['result'+str(treshold)])
        trade_cnt = np.sum((profitdf.proba0 >= treshold)&(profitdf.sl>=atr*minslratio))
        performance = overall/trade_cnt*100 if trade_cnt>0 else 0
        if (overall >= 0) & (trade_cnt > trade_cnt_th) :
            res[strname+str(treshold)] = str('{0:.1f}'.format(overall)).replace('.',',')
            res[strname+str(treshold)+'i'] = str('{0:.2f}/{1:.0f}/{2:.0f}/{3:.1f}'.format(div(overall,loss),loss,trade_cnt,tecal))
        else:
            res[strname+str(treshold)] = '0'
            res[strname+str(treshold)+'i'] = str('{0:.2f}/{1:.0f}/{2:.0f}/{3:.1f}'.format(div(overall,loss),loss,trade_cnt,tecal))
            
    return res,profitdf

def ExamineProfit2(masterframe, Xintex, y_test, proba0, proba1, trade_cnt_th,tecal,atr):
# BUY    
#     b. predict = 0 - NO trade
#     predict = 1 - trade
#     sl = close - low
#     prediction OK  profit =  high - close
#     prediction BAD  loss = sl if nextlow<low  else close - nextclose

#     atr = masterframe.atr14tr.mean()
    minslratio = 0.05 
    strname = 'S2_'
    profitdf = pd.DataFrame({'id':(Xintex).astype(int),'y_test':y_test,'proba0':proba0,'proba1':proba1})
    profitdf = profitdf.set_index(profitdf.id)
    profitdf = profitdf.drop(['id'],1)
    profitdf['low'] = masterframe.low
    profitdf['close'] = masterframe.close
    profitdf['high'] = masterframe.high
    profitdf['nextclose'] = masterframe.close.shift(-1)
    profitdf['nextlow'] = masterframe.low.shift(-1)
    profitdf['nexthigh'] = masterframe.high.shift(-1)
    
    profitdf['sl'] = masterframe.close - masterframe.low
    profitdf['tp'] =  np.where(profitdf.nexthigh >= profitdf.high,profitdf.high - profitdf.close,profitdf.nextclose-profitdf.close)
    profitdf['tp'] = np.where(profitdf.tp>atr,atr,profitdf.tp)
    profitdf['result'] = np.where(profitdf.nextlow <= profitdf.low,-1 * profitdf.sl,profitdf.tp)
    res = {}
    for treshold in [0.5,0.6,0.7,0.8,0.9]:
        profitdf['result'+str(treshold)] = profitdf.result.iloc[np.where((profitdf.proba1 >= treshold)
                                                                            &(profitdf.sl>=atr*minslratio),True,False)]
        profitdf['result'+str(treshold)] = profitdf['result'+str(treshold)] / profitdf.sl
        loss = np.sum(profitdf['result'+str(treshold)]==-1)
        overall  = np.sum(profitdf['result'+str(treshold)])
        trade_cnt = np.sum((profitdf.proba1 >= treshold)&(profitdf.sl>=atr*minslratio))
        performance = overall/trade_cnt*100 if trade_cnt>0 else 0
        if (overall >= 0) & (trade_cnt > trade_cnt_th) :
            res[strname+str(treshold)] = str('{0:.1f}'.format(overall)).replace('.',',')
            res[strname+str(treshold)+'i'] = str('{0:.2f}/{1:.0f}/{2:.0f}/{3:.1f}'.format(div(overall,loss),loss,trade_cnt,tecal))
        else:
            res[strname+str(treshold)] = '0'
            res[strname+str(treshold)+'i'] = str('{0:.2f}/{1:.0f}/{2:.0f}/{3:.1f}'.format(div(overall,loss),loss,trade_cnt,tecal))
            
    return res,profitdf

def ExamineProfit3(masterframe, Xintex, y_test, proba0, proba1, trade_cnt_th,tecal,atr):
# BUY    
#     b. predict = 0 - NO trade
#     predict = 1 - trade
#     sl = close - low
#     prediction OK  profit =  nextclose - close
#     prediction BAD  loss = sl if nextlow<low  else close - nextclose

#     atr = masterframe.atr14tr.mean()
    minslratio = 0.05 
    strname = 'S3_'
    profitdf = pd.DataFrame({'id':(Xintex).astype(int),'y_test':y_test,'proba0':proba0,'proba1':proba1})
    profitdf = profitdf.set_index(profitdf.id)
    profitdf = profitdf.drop(['id'],1)
    profitdf['low'] = masterframe.low
    profitdf['close'] = masterframe.close
    profitdf['high'] = masterframe.high
    profitdf['nextclose'] = masterframe.close.shift(-1)
    profitdf['nextlow'] = masterframe.low.shift(-1)
    profitdf['nexthigh'] = masterframe.high.shift(-1)
    
    profitdf['sl'] = masterframe.close - masterframe.low
    profitdf['tp'] = profitdf.nextclose-profitdf.close
    profitdf['tp'] = np.where(profitdf.tp>atr,atr,profitdf.tp)
    profitdf['result'] = np.where(profitdf.nextlow <= profitdf.low,-1 * profitdf.sl,profitdf.tp)
    res = {}
    for treshold in [0.5,0.6,0.7,0.8,0.9]:
        profitdf['result'+str(treshold)] = profitdf.result.iloc[np.where((profitdf.proba1 >= treshold)
                                                                            &(profitdf.sl>=atr*minslratio),True,False)]
        profitdf['result'+str(treshold)] = profitdf['result'+str(treshold)] / profitdf.sl
        loss = np.sum(profitdf['result'+str(treshold)]==-1)
        overall  = np.sum(profitdf['result'+str(treshold)])
        trade_cnt = np.sum((profitdf.proba1>=treshold) & (profitdf.sl>=atr*minslratio) )
        performance = overall/trade_cnt*100 if trade_cnt>0 else 0
        if (overall >= 0) & (trade_cnt > trade_cnt_th) :
            res[strname+str(treshold)] = str('{0:.1f}'.format(overall)).replace('.',',')
            res[strname+str(treshold)+'i'] = str('{0:.2f}/{1:.0f}/{2:.0f}/{3:.1f}'.format(div(overall,loss),loss,trade_cnt,tecal))
        else:
            res[strname+str(treshold)] = '0'
            res[strname+str(treshold)+'i'] = str('{0:.2f}/{1:.0f}/{2:.0f}/{3:.1f}'.format(div(overall,loss),loss,trade_cnt,tecal))
            
    return res,profitdf

def ExamineProfit4(masterframe, Xintex, y_test, proba0, proba1, trade_cnt_th,tecal,atr):
# BUY    
#     b. predict = 0 - NO trade
#     predict = 1 - trade
#     sl = close - low
#     prediction OK  profit =  close+(close-low)-close
#     prediction BAD  loss = sl if nextlow<low  else close - nextclose

#     atr = masterframe.atr14tr.mean()
    minslratio = 0.05 
    strname = 'S4_'
    profitdf = pd.DataFrame({'id':(Xintex).astype(int),'y_test':y_test,'proba0':proba0,'proba1':proba1})
    profitdf = profitdf.set_index(profitdf.id)
    profitdf = profitdf.drop(['id'],1)
    profitdf['low'] = masterframe.low
    profitdf['close'] = masterframe.close
    profitdf['high'] = masterframe.high
    profitdf['nextlow'] = masterframe.low.shift(-1)
    profitdf['nextclose'] = masterframe.close.shift(-1)
    profitdf['nexthigh'] = masterframe.high.shift(-1)
    
    profitdf['en_p'] = masterframe.close

    profitdf['sl_p'] = masterframe.low
    profitdf['tp_p'] = masterframe.close +(masterframe.close-masterframe.low) 
    profitdf['ee_p'] = masterframe.close.shift(-1)

    profitdf['sl_v'] = profitdf.en_p - profitdf.sl_p
    profitdf['tp_v'] = profitdf.tp_p - profitdf.en_p 
    profitdf['tp_v'] = np.where(profitdf.tp_v>atr,atr,profitdf.tp_v) #ogranicznik atr
    profitdf['ee_v'] = profitdf.ee_p - profitdf.en_p 
    profitdf['ee_v'] = np.where(profitdf.ee_v>atr,atr,profitdf.ee_v) #ogranicznik atr
    
    profitdf['tp'] = np.where(profitdf.nexthigh >= profitdf.tp_p,profitdf.tp_v,profitdf.ee_v) # albo tp albo ee
    profitdf['result'] = np.where(profitdf.nextlow <= profitdf.sl_p,-1 * profitdf.sl_v,profitdf.tp) # albo sl albo tp

    res = {}
    for treshold in [0.5,0.6,0.7,0.8,0.9]:
        profitdf['result'+str(treshold)] = profitdf.result.iloc[np.where((profitdf.proba1 >= treshold)
                                                                            &(profitdf.sl_v>=atr*minslratio),True,False)]
        profitdf['result'+str(treshold)] = profitdf['result'+str(treshold)] / profitdf.sl_v
        loss = np.sum(profitdf['result'+str(treshold)]==-1)
        overall  = np.sum(profitdf['result'+str(treshold)])
        trade_cnt = np.sum((profitdf.proba1 >= treshold)&(profitdf.sl_v>=atr*minslratio))
        performance = overall/trade_cnt*100 if trade_cnt>0 else 0
        if (overall >= 0) & (trade_cnt > trade_cnt_th) :
            res[strname+str(treshold)] = str('{0:.1f}'.format(overall)).replace('.',',')
            res[strname+str(treshold)+'i'] = str('{0:.2f}/{1:.0f}/{2:.0f}/{3:.1f}'.format(div(overall,loss),loss,trade_cnt,tecal))
        else:
            res[strname+str(treshold)] = '0'
            res[strname+str(treshold)+'i'] = str('{0:.2f}/{1:.0f}/{2:.0f}/{3:.1f}'.format(div(overall,loss),loss,trade_cnt,tecal))
            
    return res,profitdf

def div(a,b):
    if b!=0:
        return a/b
    else:
        return 0
    