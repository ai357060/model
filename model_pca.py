import pandas as pd
import numpy as np
import time
import importlib
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

import model_collection
from model_collection import *

pd.options.display.max_columns = None

def runhypermodel(featsel='pca'):

    fver = 'v13'
    masterframe = loaddata_master('../Data/mf_UJ1440_'+fver+'.csv')

    # Prepare Y
    Rtp=1
    Rsl=1
    masterframe['y'] = -1
    n = 5
    i = 0
    while i < len(masterframe) - n:   
        j = 1
        yy = False
        while j <= n:
            #if (masterframe.low.iloc[i+j] < masterframe.close.iloc[i]-Rsl*masterframe.atr14atr.iloc[i]):
            if (masterframe.low.iloc[i+j] < masterframe.low.iloc[i]-Rsl*masterframe.atr14atr.iloc[i]):
                yy = False
                break
            #if (masterframe.high.iloc[i+j] > masterframe.close.iloc[i]+Rtp*masterframe.atr14atr.iloc[i]):
            if (masterframe.high.iloc[i+j] > masterframe.high.iloc[i]+Rtp*masterframe.atr14atr.iloc[i]):
                yy = True
                break
            j = j + 1

        if yy == True:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 1            
            #masterframe.iloc[i+1:i+j+1,masterframe.columns.get_loc('y')]=0      #nochain
            #i = i + j                                                           #nochain 
            i = i + 1   #chain

        else:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 0
            i = i + 1
        
    '''      
    # Prepare Y
    Rtp=0
    Rsl=0
    masterframe['y'] = -1
    n = 1
    i = 0
    while i < len(masterframe) - n:   
        j = 1
        yy = False
        while j <= n:
            if (masterframe.high.iloc[i+j] > masterframe.high.iloc[i]):
                yy = True
                break
            j = j + 1

        if yy == True:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 1            
            #masterframe.iloc[i+1:i+j+1,masterframe.columns.get_loc('y')]=0      #nochain
            #i = i + j                                                           #nochain 
            i = i + 1   #chain

        else:
            masterframe.iloc[i,masterframe.columns.get_loc('y')] = 0
            i = i + 1
    '''

    #pre-prepare data
    orygframe = masterframe.copy()
    masterframe = masterframe.drop(['volume'],1)
    masterframe.dropna(inplace=True)

    # split data
    # masterframe = masterframe[-3600:] ###testowo
    X_df = masterframe.iloc[:-1, 2:-1] 
    y_df = masterframe.iloc[:-1, -1] 
    featurenames = masterframe.iloc[:-1, 2:-1].columns.values

    X = X_df.values
    y = y_df.values
    y = y.astype('int')
    X = X.astype('float')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25, shuffle = False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.25, shuffle = False)

    # balance classes
    from imblearn.over_sampling import SMOTE
    # from imblearn.over_sampling import ADASYN
    # from imblearn.over_sampling import BorderlineSMOTE
    # from imblearn.over_sampling import RandomOverSampler
    # from imblearn.over_sampling import SVMSMOTE
    sm = SMOTE(random_state=27)
    # sm = ADASYN(random_state=27)
    # sm = BorderlineSMOTE(random_state=27)
    # sm = RandomOverSampler(random_state=27)
    # sm = SVMSMOTE(random_state=27)
    X_train, y_train = sm.fit_sample(X_train, y_train)

    # X_test_df = pd.DataFrame(X_test)
    # X_test_df.to_csv(sep=';',path_or_buf='../Data/x_pre.csv',date_format="%Y-%m-%d",index = False)
    
    #Scale
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    modeltype = {}
    featcount = []
    testone=False
    # featsel = 'rf'
    # featsel = 'svc'
    # featsel = 'pca'
    # featsel = 'all'
    featcount.append(5);modeltype[5]   = ['rf','svc','mlp']
    # featcount.append(10);modeltype[10] = ['rf','svc','mlp']
    featcount.append(15);modeltype[15] = ['rf','svc','mlp']
    # featcount.append(20);modeltype[20] = ['rf','svc','mlp']
    featcount.append(25);modeltype[25] = ['rf','svc','mlp']    
    

    if featsel != 'all':
        for i in featcount:
            print('FEATSEL:'+featsel+str(i)+'________________________________________________________________________________________')

            if featsel == 'rf':
                select = RFE(RandomForestClassifier(n_estimators=100,random_state=2,n_jobs=1),n_features_to_select=i)
                select.fit(X_train, y_train)
                X_train_rfe= select.transform(X_train) 
                X_test_rfe= select.transform(X_test)
                X_train_sc_rfe= select.transform(X_train_sc)  
                X_test_sc_rfe= select.transform(X_test_sc)    

            if featsel == 'svc':
                select = RFE(SVC(kernel='linear')            ,n_features_to_select=i)
                select.fit(X_train_sc, y_train)
                X_train_rfe= select.transform(X_train) 
                X_test_rfe= select.transform(X_test)
                X_train_sc_rfe= select.transform(X_train_sc)  
                X_test_sc_rfe= select.transform(X_test_sc)  

            if featsel == 'pca':
                select = PCA(n_components=i, whiten=True, random_state=2)
                select.fit(X_train)
                X_train_rfe= select.transform(X_train) 
                X_test_rfe= select.transform(X_test)
                X_train_sc_rfe= X_train_rfe
                X_test_sc_rfe= X_test_rfe

        #     select = PCA(n_components=i, whiten=False, random_state=2)
        #     select.fit(X_train_sc)
        #     X_train_rfe= select.transform(X_train) 
        #     X_test_rfe= select.transform(X_test)
        #     X_train_sc_rfe= select.transform(X_train_sc)
        #     X_test_sc_rfe= select.transform(X_test_sc)
        #     featsel = 'pca_nw'

            if testone == True:
                featsel = 'temp_'+featsel

            # visualize the selected features:
            #mask = select.get_support()
            #plt.matshow(mask.reshape(1, -1), cmap='gray_r')
            #plt.xlabel("Sample index")
            #print(X_df.iloc[:2,mask])
            #print("Test score: {:.3f}".format(select.score(X_test_sc, y_test)))
            #print("Test score: {:.3f}".format(select.score(X_test, y_test)))

            #lin_resdf = ExamineLogisticRegression(orygframe,X_test[:,0],X_train_rfe, y_train,X_test_rfe, y_test,featurenames,testone=False,plot=False)
            #lin_resdf.to_csv(sep=';',path_or_buf='Resu/'+fver+'_'+featsel+str(i)+'_LogisticRegression'+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)

            #lin_resdf = ExamineLinearSVC(orygframe,X_test[:,0],X_train_rfe, y_train,X_test_rfe, y_test,featurenames,testone=False,plot=False)
            #lin_resdf.to_csv(sep=';',path_or_buf='Resu/'+fver+'_'+featsel+str(i)+'_LinearSVC'+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)


            if 'rf' in modeltype[i]:
                print('FEATSEL:'+featsel+str(i)+'_model_rf___________________________________________________________________________')
                forest_resdf = ExamineRandomForest(orygframe, X_test[:,0], X_train_rfe, y_train,X_test_rfe, y_test, featurenames, testone=testone, plot=False, automaxfeat=True)
                forest_resdf.to_csv(sep=';', path_or_buf='../Resu/'+fver+'_'+featsel+str(i)+'_RandomForest'+str(int(time.time()))+'.csv', date_format="%Y-%m-%d", index = False)

            if 'svc' in modeltype[i]:
                print('FEATSEL:'+featsel+str(i)+'_model_svc___________________________________________________________________________')
                svc_resdf = ExamineSVC(orygframe,X_test[:,0], X_train_sc_rfe, y_train, X_test_sc_rfe, y_test,featurenames, testone=testone, plot=False)
                svc_resdf.to_csv(sep=';', path_or_buf='../Resu/'+fver+'_'+featsel+str(i)+'_SVC'+str(int(time.time()))+'.csv', date_format="%Y-%m-%d", index = False)

            if 'mlp' in modeltype[i]:
                print('FEATSEL:'+featsel+str(i)+'_model_mlp___________________________________________________________________________')
                mlp_resdf = ExamineMLP(orygframe,X_test[:,0],X_train_sc_rfe, y_train, X_test_sc_rfe, y_test, featurenames, testone=testone, plot=False)
                mlp_resdf.to_csv(sep=';', path_or_buf='../Resu/'+fver+'_'+featsel+str(i)+'_MLP'+str(int(time.time()))+'.csv', date_format="%Y-%m-%d", index = False)

        print('FEATSEL________________finished________________________________________________________________________________')

        
    if featsel == 'all':
        print('ALL_____________rf___________________________________________________________________________')
        featsel = 'all_RandomForest' if testone == False else 'temp_all_RandomForest'
        forest_resdf = ExamineRandomForest(orygframe,X_test[:,0],X_train, y_train,X_test, y_test,featurenames,testone=testone,plot=False,automaxfeat=False)
        forest_resdf.to_csv(sep=';',path_or_buf='../Resu/'+fver+'_'+featsel+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)

        print('ALL_____________svc___________________________________________________________________________')
        featsel = 'all_SVC' if testone == False else 'temp_all_SVC'
        svc_resdf = ExamineSVC(orygframe,X_test[:,0],X_train_sc, y_train,X_test_sc, y_test,featurenames,testone=testone,plot=False)
        svc_resdf.to_csv(sep=';',path_or_buf='../Resu/'+fver+'_'+featsel+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)

        print('ALL_____________mlp___________________________________________________________________________')
        featsel = 'all_MLP' if testone == False else 'temp_all_MLP'
        mlp_resdf = ExamineMLP(orygframe,X_test[:,0],X_train_sc, y_train,X_test_sc, y_test,featurenames,testone=testone,plot=False)
        mlp_resdf.to_csv(sep=';',path_or_buf='../Resu/'+fver+'_'+featsel+str(int(time.time()))+'.csv',date_format="%Y-%m-%d",index = False)

        print('ALL________________finished________________________________________________________________________________')
    
    return
