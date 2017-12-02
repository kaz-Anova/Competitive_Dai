# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 16:24:41 2017

@author: mimar
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
import lightgbm as lgb

def bagged_set(X_ts,y_cs, seed, estimators, xt,yt=None):
    
   # create array object to hold predictions 
   baggedpred=np.array([ 0.0 for d in range(0, xt.shape[0])]) 

   #loop for as many times as we want bags
   for n in range (0, estimators):
       
       params = {	'objective': 'huber',
                 
                'metric': 'auc',
                'boosting': 'gbdt',
                'learning_rate': 0.02, #change here    
                'huber_delta':0.8,
                'verbose': 0,    
                'num_leaves': 25, # ~18    
                'bagging_fraction': 0.9, 
                #'categorical_feature':'2,4,5,22,23,24,25,26,27,28,29,30,31,32',                
                'bagging_freq': 1,    
                'bagging_seed': seed + n,    
                'feature_fraction': 0.4,    
                'feature_fraction_seed': seed + n,    
                'min_data_in_leaf': 25, #30, #56, # 10-50    
                'max_bin': 255, # maybe useful with overfit problem    
                'max_depth':12,    
                #'reg_lambda': 10,    
                'reg_alpha':2,    
                'lambda_l2': 2.5,
                'num_threads':18
                }

       d_train = lgb.Dataset(X_ts,y_cs, free_raw_data=False)#np.log1p(
       if type(yt)!=type(None):           
           d_cv = lgb.Dataset(xt,yt, free_raw_data=False, reference=d_train)#, reference=d_train
           model = lgb.train(params,d_train,num_boost_round=600,
                             valid_sets=d_cv,

                             verbose_eval=True ) #1000                        
           
       else :
           d_cv = lgb.Dataset(xt, free_raw_data=False)  
           model = lgb.train(params,d_train,num_boost_round=600) #1000                        

       preds=model.predict(xt)                
       # update bag's array
       baggedpred+=preds
            
       print("completed: " + str(n)  )                 

   # divide with number of bags to create an average estimate  
   baggedpred/= estimators
     
   return baggedpred

#train = np.loadtxt('first_level_train.csv')
#test = np.loadtxt('first_level_test.csv')

#train=np.loadtxt('train.csv',usecols=[k for k in range(1,9)], skiprows=1, delimiter=",")
#test = np.loadtxt('test.csv',usecols=[k for k in range(1,9)], skiprows=1, delimiter=",")

X=np.loadtxt('train_munged.csv',usecols=[k for k in range(0,55)], skiprows=1, delimiter=",")
y =np.loadtxt("train.csv", delimiter=',',usecols=[0], skiprows=1)


meta_folder="meta_folder/"
if not os.path.exists(meta_folder):      #if it does not exists, we create it
    os.makedirs(meta_folder)  
    
output_name="lightgbmreg_script_v2"   
print(X.shape, y.shape)           
 
############### Params section #####################
bagging=15 # number of models trained with different seeds
number_of_folds = 5  # number of folds in strattified cv
kfolder=StratifiedKFold(y, n_folds= number_of_folds,shuffle=True, random_state=1)           
#model to use
#modelextra= ExtraTreesClassifier(n_estimators=1000, criterion='entropy', max_depth=40, 
                            #min_samples_split=4,min_samples_leaf=2, max_features=0.7,n_jobs=25, random_state=1)
train_stacker=[ 0.0  for k in range (0,X.shape[0]) ]

#create target variable        
mean_kapa = 0.0
kfolder=StratifiedKFold(y, n_folds=number_of_folds,shuffle=True, random_state=1)
#X,y=shuffle(X,y, random_state=SEED) # Shuffle since the data is ordered by time

i=0 # iterator counter
print ("starting cross validation with %d kfolds " % (number_of_folds))
for train_index, test_index in kfolder:
    # creaning and validation sets
    X_train, X_cv = X[train_index], X[test_index]
    y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
    print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))

    preds= bagged_set(X_train,y_train, 1, bagging, X_cv,yt=None     )# y_cv
              
    auc=roc_auc_score(y_cv, preds)
    print ("size train: %d size cv: %d auc (fold %d/%d): %f" % ((X_train.shape[0]), (X_cv.shape[0]), i + 1, number_of_folds,auc))
 
    mean_kapa += auc
    #save the results
    no=0
    for real_index in test_index:
             train_stacker[real_index]=(preds[no])
             no+=1
    i+=1 # increment cv iterator
   
if (number_of_folds)>0:
    mean_kapa/=number_of_folds
    print (" Average auc : %f" % (mean_kapa) )
    np.savetxt(meta_folder+output_name+ ".train.csv",np.array(train_stacker), fmt='%.6f') 
print (" printing train datasets ")


X_test = np.loadtxt('test_munged.csv',usecols=[k for k in range(0,55)], skiprows=1, delimiter=",")
preds= bagged_set(X,y, 1, bagging, X_test,yt=None)
# === Predictions === #
print (" printing test datasets ")        
np.savetxt(meta_folder+output_name+ ".test.csv",np.array(preds), fmt='%.6f')        


print("Write results...")
output_file = "submission_"+str( (mean_kapa ))+".csv"
print("Writing submission to %s" % output_file)
f = open(output_file, "w")   
f.write("id,action\n")# the header   
for g in range(0, len(preds))  :
  f.write("%d,%f\n" % (g+1,preds[g]))
f.close()
print("Done.")  




 
