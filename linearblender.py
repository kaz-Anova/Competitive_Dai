 # -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score,r2_score,f1_score,matthews_corrcoef,mean_absolute_error,mean_squared_error,log_loss,precision_score,recall_score
import numpy as np



class linearblender(BaseEstimator, ClassifierMixin):
  def __init__(self,
               task="regression",
               metric=None,
               n_classes=None,                
               n_iter=5,
               precision="simple",                
               verbose=0 ):   
      
    assert task in ['regression', 'classification']
    assert metric in [None, 'auc','r2','mae','rmse','logloss']
    if type(metric)==type(None):
        if task=='regression':
            metric='rmse'
        else :
            metric='logloss'            

    assert type(n_classes) is type(None) or type(n_classes) is int
    if type(n_classes) != type(None):
        assert n_classes>1
       
    assert type(n_iter) is int and n_iter>0   
    assert type(verbose) is int    
    assert precision in ['simple', 'medium','exhaustive']
    assert metric in [ 'auc','r2','f1','matthews','mae','rmse','logloss','precision','recall']
    if metric in [ 'auc','f1','matthews','logloss','precision','recall'] and task!="classification":
        raise Exception (" the metric selected")
        
    self.metric=metric   
    ## the metric lists that uses 'predict' when computed the metric
    self.predict_metrics=['r2','f1','matthews','mae','rmse','precision','recall']
    ## the metric lists that uses 'predict_proba' when computed the metric    
    self.predict_proba_metrics=['auc','logloss']
    # metrics which the higher they are. the better it is 
    self.high_is_good=[ 'auc','r2','f1','matthews','precision','recall']
    # metrics which the lower they are. the better it is 
    self.low_is_good=[ 'mae','rmse','logloss']    
    self.n_models=0
    
    self.metric=metric
    self.verbose= verbose
    self.precision= precision
    self.n_classes= n_classes
    
    self.task=task   
    self.n_iter=n_iter
    self.weights=[]
    self.increment=0.2
    self.rounds=5
    if precision=="medium":
        self.increment=0.1
        self.rounds=10
    elif precision=="exhaustive":
        self.increment=0.05    
        self.rounds=20
    
     

  """
  actual: actual values
  pred : predicted values
  return: the metric value
  """
  def compute_metric(self,actual, pred):
      if self.metric=='auc':
          return roc_auc_score(actual, pred)
      elif self.metric=='r2':
          return r2_score(actual, pred)
      elif self.metric=='f1':
          return f1_score(actual, pred)
      elif self.metric=='matthews':
          return matthews_corrcoef(actual, pred)  
      elif self.metric=='mae':
          return mean_absolute_error(actual, pred)        
      elif self.metric=='mae':
          return mean_absolute_error(actual, pred)    
      elif self.metric=='rmse':
          return np.sqrt(mean_squared_error(actual, pred)  )      
      elif self.metric=='logloss':
          return log_loss(actual, pred) 
      elif self.metric=='precision':
          return precision_score(actual, pred) 
      elif self.metric=='recall':
          return recall_score(actual, pred)
      else :
        raise(" your metric is not recognised") 


  def scale_weights(self, w ):
      sum_w=sum(w)  
      if sum_w!=0.0:
          ws=[wei/sum_w for wei in w]
      else :
          ws=w[:]
      return ws
      
      
  """
  X : 2-d numpy array
  y : 1-d numpy array
  """ 
  
  def fit(self, X, y ):  
      
      if type(X) == type(None) or (not type(X) is np.ndarray and isinstance(X[0], list)==False):
          raise Exception (" X needs to be a numpy array or a 2d list")
      if not type(y) is np.ndarray and not type(y) is list :
           raise Exception (" y needs to be a numpy array or list")
      
      if not isinstance(X[0], list):
          X=X.tolist()
      
      # determine number of models
      if self.task=="regression":
         self.n_models=len(X[0])
      elif  self.task=="classification":
          columns_shape=len(X[0])
          if self.n_classes==2:
               self.n_models=columns_shape
          elif  self.n_classes>2:
              remaining=columns_shape%self.n_classes
              if remaining!=0:
                   raise Exception (" dividing columns with n_classes,leaves a remainder...")
              else :
                  self.n_models=columns_shape/self.n_classes
      if  self.n_models<1:
           raise Exception ("models needs to be 1 or more")

      self.weights=[0.0 for k in range (self.n_models)]
      
      avg_metric=-9999999.99
      if self.metric in self.low_is_good:
          avg_metric=9999999.99

      for e in range ( self.n_iter):
          if self.verbose>1:
              print (" Linear blending iter: %d" % (e+1))
          for c in range (self.n_models):
              temp_weights=[ k for k in self.weights]
              if self.verbose>1:
                  print (" Linear blending iter %d examining model %d : " % (e+1,c))              
              for r in range (self.rounds): 
                  if self.verbose>1:
                      print (" Linear blending iter %d examining model %d weight %f : " % (e+1,c,(r+1)* self.increment))                   
                  temp_weights[c]=(r+1)* self.increment
                  temp_weights_scaled=self.scale_weights(temp_weights)
                  if self.task=="regression" or self.n_classes<=2:
                      preds=[0.0 for k in range (len(X))]
                      for a in range (len(X)):
                          for jj in range (len(temp_weights_scaled)):
                            preds[a]+=X[a][jj]*temp_weights_scaled[jj]
                  else :
                      preds=[ [0.0 for ss in range(self.n_classes)]  for k in range (len(X))]
                      #print (temp_weights_scaled)
                      for a in range (len(X)):
                          for jj in range (len(temp_weights_scaled)):
                              for jjj in range (self.n_classes):                              
                                  preds[a][jjj]+=X[a][jj*self.n_classes+ jjj]*temp_weights_scaled[jj]  
                          bestf=0.0
                          for asd in range(self.n_classes):
                              bestf+=preds[a][asd]
                          for asd in range(self.n_classes):
                              preds[a][asd]/=bestf                                  
                                  
                        
                  this_metric=self.compute_metric(y,preds)  
                  if self.metric in self.low_is_good and this_metric<avg_metric:
                      if self.verbose>0:
                          print (" Linear blending iter %d examining model %d weight %f , metric %s improved from %f to %f : " % (e+1,c,(r+1)* self.increment,self.metric,avg_metric,this_metric ))                                        
                      
                      avg_metric=this_metric
                      self.weights[c]=(r+1)* self.increment
                  elif self.metric in self.high_is_good and this_metric>avg_metric:
                      if self.verbose>0:
                          print (" Linear blending iter %d examining model %d weight %f , metric %s improved from %f to %f : " % (e+1,c,(r+1)* self.increment,self.metric,avg_metric,this_metric )   )                                                                                 
                      avg_metric=this_metric
                      self.weights[c]=(r+1)* self.increment 
                  else :
                      if self.verbose>1:
                          print (" Linear blending iter %d examining model %d weight %f , metric %s There is NO improvement from %f (to %f) : " % (e+1,c,(r+1)* self.increment,self.metric,avg_metric,this_metric )   )                                                                                 
                                               
      self.weights=self.scale_weights(self.weights)
      if self.verbose>0:
          print ("final weights, ",self.weights )
      
      return self

  """
  X : 2-d numpy array
  """ 
  

  def predict(self, X): 
      
      if type(X) == type(None) or (not type(X) is np.ndarray and isinstance(X[0], list)==False):
          raise Exception (" X needs to be a numpy array or a 2d list")
      if len(self.weights)<1:
          raise Exception (" fit method needs to run successfuly prior to calling predict() ")  
      if not isinstance(X[0], list):
          X=X.tolist()
      if self.task=="regression" or self.n_classes<=2:
          preds=[0.0 for k in range (len(X))]
          for a in range (len(X)):
              for jj in range (len(self.weights)):
                preds[a]+=X[a][jj]*self.weights[jj] 
              if self.task=="classificaion":
                  if preds[a]>=0.5:
                      preds[a]=1.
                  else :
                      preds[a]=0.     
      else :
          probs=[ [0.0 for ss in range(self.n_classes)]  for k in range (len(X))]
          preds=[0.0 for k in range (len(X))]
          for a in range (len(X)):
              for jj in range (len(self.weights)):
                  for jjj in range (self.n_classes):                              
                      probs[a][jjj]+=X[a][jj*self.n_classes+ jjj]*self.weights[jj]    
              maxes=probs[a][0]
              bestf=0
              for asd in range(1,self.n_classes):
                  if probs[a][asd]>maxes:
                      maxes=probs[a][asd]
                      bestf=asd
              preds[a]= bestf
                  
      return np.array(preds)
  
  """
  n_models :number of models to set linear weights
  """                             
  
  def set_linear_blend(self,n_models): 
       self.weights=[1.0 for s in range (n_models)]  
       self.scale_weights(self.weights)
       
  """
  n_classes : number of classess to set
  """                             
  
  def set_n_classes(self,n_classes): 
       self.n_classes=n_classes       
  """
  X : 2-d numpy array
  """           
  
  def predict_proba(self, X): 
      
      if type(X) == type(None) or (not type(X) is np.ndarray and isinstance(X[0], list)==False):
          raise Exception (" X needs to be a numpy array or a 2d list")
      if len(self.weights)<1:
          raise Exception (" fit method needs to run successfuly prior to calling predict() ")  
      if not isinstance(X[0], list):
          X=X.tolist()
          
      if self.task=="regression":
          
          preds=[0.0 for k in range (len(X))]
          for a in range (len(X)):
              for jj in range (len(self.weights)):
                preds[a]+=X[a][jj]*self.weights[jj]       
 
      elif self.n_classes<=2:
          preds=[ [0.0 for k in range(2)] for k in range (len(X))]
          for a in range (len(X)):
              for jj in range (len(self.weights)):
                preds[a][1]+=X[a][jj]*self.weights[jj]
              preds[a][0]=1.-preds[a][1]
    
      else :
          preds=[ [0.0 for ss in range(self.n_classes)]  for k in range (len(X))]
          for a in range (len(X)):
              for jj in range (len(self.weights)):
                  for jjj in range (self.n_classes):                              
                      preds[a][jjj]+=X[a][jj*self.n_classes+ jjj]*self.weights[jj] 
              bestf=0.0
              for asd in range(self.n_classes):
                  bestf+=preds[a][asd]
              for asd in range(self.n_classes):
                  preds[a][asd]/=bestf
                  
      return np.array(preds)