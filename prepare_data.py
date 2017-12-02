

#amazon helper functions

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix,csc_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib



"""
converts sparse data to StackNet format
Better use this one than standard svmlight.

"""
def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):    
    zsparse=csr_matrix(csc_matrix(array))
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))    
    print(" indptr lenth %d" % (len(indptr)))
    
    f=open(filename,"w")
    counter_row=0
    for b in range(0,len(indptr)-1):
        #if there is a target, print it else , print nothing
        if ytarget!=None:
             f.write(str(ytarget[b]) + deli1)     
             
        for k in range(indptr[b],indptr[b+1]):
            if (k==indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k],deli2,-1))
                else :
                    f.write("%d%s%f" % (indices[k],deli2,data[k]))                    
            else :
                if np.isnan(data[k]):
                     f.write("%s%d%s%f" % (deli1,indices[k],deli2,-1))  
                else :
                    f.write("%s%d%s%f" % (deli1,indices[k],deli2,data[k]))
        f.write("\n")
        counter_row+=1
        if counter_row%10000==0:    
            print(" row : %d " % (counter_row))    
    f.close()  
    
"""
Load training and test data. Then create in a brute force way to cerate all possible 5-way 
categorical interractions and test whether auc improves when adding them. 
Once it finds the best interractions, it prints them as sparse data
as:
    train.sparse
    test.sparse
"""
  
def create_5way_interractions(path=""):
    
    train_df=pd.read_csv(path + "train.csv")
    test_df=pd.read_csv(path + "test.csv")
    train_df.drop("ROLE_CODE", axis=1, inplace=True)
    test_df.drop("ROLE_CODE", axis=1, inplace=True)
    
    y=np.array(train_df['ACTION'])
    train_df.drop("ACTION", axis=1, inplace=True)
    test_df.drop("id", axis=1, inplace=True)  
    
    columns=train_df.columns.values
    columns=[columns[k] for k in range(0,len(columns))] # we exclude the first column
    
    kfolder=StratifiedKFold(y, n_folds=5,shuffle=True, random_state=1) 
    
    grand_auc=0
    
    X=np.array(train_df)
    #X,y=shuffle(X,y, random_state=SEED) # Shuffle since the data is ordered by time
    i=0 # iterator counter
    model=SGDClassifier(loss='log', penalty='l2', alpha=0.0000225, n_iter=50, random_state=1)
    for train_index, test_index in kfolder:    
            X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
            one=OneHotEncoder(handle_unknown='ignore')
            one.fit(X_train)
            X_train=one.transform(X_train)
            X_cv=one.transform(X_cv) 
            model.fit(X_train,y_train)
            preds=model.predict_proba(X_cv)[:,1]
            auc=roc_auc_score(y_cv,preds)
            print (" fold %d/%d auc %f " % (i+1,5,auc))
            grand_auc+=auc
            i+=1
    grand_auc/=5
    print ("grand AUC is %f " % (grand_auc))
    
    columns=train_df.columns.values
    columns=[columns[k] for k in range(0,len(columns))] # we exclude the first column
    cols=[k for k in columns]
    newcols=cols[:]
    print(cols)
    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
                name1=columns[j1] + "_plus_" + columns[j2]
                cols.append(name1)

                train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))
                test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) 
                lbl = LabelEncoder()
                lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                train_df[name1] = lbl.transform(list(train_df[name1].values))
                test_df[name1] = lbl.transform(list(test_df[name1].values))                
                
                mean_auc=0
                X=np.array(train_df)
                i=0 # iterator counter    
                for train_index, test_index in kfolder:    
                        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
                        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
                        one=OneHotEncoder(handle_unknown='ignore')
                        one.fit(X_train)
                        X_train=one.transform(X_train)
                        X_cv=one.transform(X_cv) 
                        model.fit(X_train,y_train)
                        preds=model.predict_proba(X_cv)[:,1]
                        auc=roc_auc_score(y_cv,preds)
                        print (" %s fold %d/%d auc %f " % (name1,i+1,5,auc))
                        mean_auc+=auc
                        i+=1
                mean_auc/=5  
                if (mean_auc>grand_auc+0.00001):
                    print (" %s will remain fold new Auc %f versus old Auc %f " % (name1,mean_auc,grand_auc))
                    grand_auc=mean_auc
                    newcols.append(name1)
                else :
                   print( "dropping %s as %f is NOT big enough to %f " %  (name1,mean_auc,grand_auc))
                   train_df.drop(name1, inplace=True,axis=1) 
                   test_df.drop(name1, inplace=True,axis=1) 
                   
                
    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
            for j3 in range(j2+1,len(columns)):            
                name1=columns[j1] + "_plus_" + columns[j2]+ "_plus_" + columns[j3]
                cols.append(name1)
                
                train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))+ "_" + train_df[columns[j3]].apply(lambda x:str(x))
                test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) + "_" + test_df[columns[j3]].apply(lambda x:str(x)) 
                lbl = LabelEncoder()
                lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                train_df[name1] = lbl.transform(list(train_df[name1].values))
                test_df[name1] = lbl.transform(list(test_df[name1].values))                       
                
                mean_auc=0
                X=np.array(train_df)
                i=0 # iterator counter    
                for train_index, test_index in kfolder:    
                        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
                        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
                        one=OneHotEncoder(handle_unknown='ignore')
                        one.fit(X_train)
                        X_train=one.transform(X_train)
                        X_cv=one.transform(X_cv) 
                        model.fit(X_train,y_train)
                        preds=model.predict_proba(X_cv)[:,1]
                        auc=roc_auc_score(y_cv,preds)
                        print (" %s fold %d/%d auc %f " % (name1,i+1,5,auc))
                        mean_auc+=auc
                        i+=1
                mean_auc/=5  
                if (mean_auc>grand_auc+0.00001):
                    print (" %s will remain fold new Auc %f versus old Auc %f " % (name1,mean_auc,grand_auc))
                    grand_auc=mean_auc
                    newcols.append(name1)
                else :
                   print( "dropping %s as %f is NOT big enough to %f " %  (name1,mean_auc,grand_auc))
                   train_df.drop(name1, inplace=True,axis=1) 
                   test_df.drop(name1, inplace=True,axis=1) 

    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
            for j3 in range(j2+1,len(columns)):            
                for j4 in range(j3+1,len(columns)):                
                    name1=columns[j1] + "_plus_" + columns[j2]+ "_plus_" + columns[j3]+ "_plus_" + columns[j4]
                    cols.append(name1)

                    train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))+ "_" + train_df[columns[j3]].apply(lambda x:str(x))+ "_" + train_df[columns[j4]].apply(lambda x:str(x))
                    test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) + "_" + test_df[columns[j3]].apply(lambda x:str(x)) + "_" + test_df[columns[j4]].apply(lambda x:str(x)) 
                    lbl = LabelEncoder()
                    lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                    train_df[name1] = lbl.transform(list(train_df[name1].values))
                    test_df[name1] = lbl.transform(list(test_df[name1].values))                
                    
                    mean_auc=0
                    X=np.array(train_df)
                    i=0 # iterator counter    
                    for train_index, test_index in kfolder:    
                            X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
                            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
                            one=OneHotEncoder(handle_unknown='ignore')
                            one.fit(X_train)
                            X_train=one.transform(X_train)
                            X_cv=one.transform(X_cv) 
                            model.fit(X_train,y_train)
                            preds=model.predict_proba(X_cv)[:,1]
                            auc=roc_auc_score(y_cv,preds)
                            print (" %s fold %d/%d auc %f " % (name1,i+1,5,auc))
                            mean_auc+=auc
                            i+=1
                    mean_auc/=5  
                    if (mean_auc>grand_auc+0.00001):
                        print (" %s will remain fold new Auc %f versus old Auc %f " % (name1,mean_auc,grand_auc))
                        grand_auc=mean_auc
                        newcols.append(name1)
                    else :
                       print( "dropping %s as %f is NOT big enough to %f " %  (name1,mean_auc,grand_auc))
                       train_df.drop(name1, inplace=True,axis=1) 
                       test_df.drop(name1, inplace=True,axis=1) 

    for j1 in range(0,len(columns)):
        for j2 in range(j1+1,len(columns)):
            for j3 in range(j2+1,len(columns)):            
                for j4 in range(j3+1,len(columns)):     
                    for j5 in range(j4+1,len(columns)):                       
                        name1=columns[j1] + "_plus_" + columns[j2]+ "_plus_" + columns[j3]+ "_plus_" + columns[j4]+ "_plus_" + columns[j5]
                        cols.append(name1)
                        
                        train_df[name1]=train_df[columns[j1]].apply(lambda x:str(x)) + "_" + train_df[columns[j2]].apply(lambda x:str(x))+ "_" + train_df[columns[j3]].apply(lambda x:str(x))+ "_" + train_df[columns[j4]].apply(lambda x:str(x))+ "_" + train_df[columns[j5]].apply(lambda x:str(x))
                        test_df[name1]=test_df[columns[j1]].apply(lambda x:str(x))+ "_" + test_df[columns[j2]].apply(lambda x:str(x)) + "_" + test_df[columns[j3]].apply(lambda x:str(x)) + "_" + test_df[columns[j4]].apply(lambda x:str(x)) + "_" + test_df[columns[j5]].apply(lambda x:str(x))
                        lbl = LabelEncoder()
                        lbl.fit(list(train_df[name1].values) + list(test_df[name1].values))
                        train_df[name1] = lbl.transform(list(train_df[name1].values))
                        test_df[name1] = lbl.transform(list(test_df[name1].values))                             
                        
                        mean_auc=0
                        X=np.array(train_df)
                        i=0 # iterator counter    
                        for train_index, test_index in kfolder:    
                                X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
                                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]     
                                one=OneHotEncoder(handle_unknown='ignore')
                                one.fit(X_train)
                                X_train=one.transform(X_train)
                                X_cv=one.transform(X_cv) 
                                model.fit(X_train,y_train)
                                preds=model.predict_proba(X_cv)[:,1]
                                auc=roc_auc_score(y_cv,preds)
                                print (" %s fold %d/%d auc %f " % (name1,i+1,5,auc))
                                mean_auc+=auc
                                i+=1
                        mean_auc/=5  
                        if (mean_auc>grand_auc+0.00001):
                            print (" %s will remain fold new Auc %f versus old Auc %f " % (name1,mean_auc,grand_auc))
                            grand_auc=mean_auc
                            newcols.append(name1)
                        else :
                           print( "dropping %s as %f is NOT big enough to %f " %  (name1,mean_auc,grand_auc))
                           train_df.drop(name1, inplace=True,axis=1) 
                           test_df.drop(name1, inplace=True,axis=1) 
                       
    train_df.to_csv("trainid.csv",index=False)
    test_df.to_csv("testid.csv",index=False) 
      
    print ("one hot encoding")
    train=np.array(train_df)
    test=np.array(test_df) 
    
    for j in range(0,train.shape[1]):
        dicter=defaultdict(lambda:0)
        for i in range(0,train.shape[0]):
           dicter[str(train[i,j])]+=1 
        for i in range(0,test.shape[0]):
           dicter[str(test[i,j])]+=1 
        for i in range(0,train.shape[0]):
          train[i,j]=9999999 if dicter[str(train[i,j])]<=1 else  train[i,j]
        for i in range(0,test.shape[0]):
           test[i,j]=9999999 if dicter[str(test[i,j])]<=1 else  test[i,j]   
          
    one=OneHotEncoder(handle_unknown='ignore', sparse=True)
    test=one.fit_transform(test)
    train=one.transform(train)   
    test=csr_matrix(test)
    train=csr_matrix(train)   
    
    joblib.dump((train,test), "sparse_sets.pkl")

       


############ code runs here############
    
create_5way_interractions() # compute 5way interractions
 
    
    
    
    