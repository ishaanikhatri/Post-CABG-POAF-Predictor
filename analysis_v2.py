# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:33:09 2020
Resurrected on Sept 1 2022

@author: Wasiq
@author: Sud
"""
import win32file
import matplotlib.pyplot as plt
import statistics
import missingpy
import scipy as sp
import IPython
import sklearn
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from missingpy import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from multiprocessing import Process
from multiprocessing import Pool

win32file._setmaxstdio(2048)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 400)
POAF=pd.read_csv("FINAL_PROCESSED_DATA_WS.csv")
POAF_X=POAF.iloc[:,0:396]
POAF_Y=POAF.iloc[:,396]
labels=POAF_X.columns.values
X_train_set, X_test_set, Y_train_set, Y_test_set=train_test_split(POAF_X, POAF_Y, stratify=POAF_Y, random_state=69)
for col in X_test_set.columns[0:50]:
    X_test_set[col]=(X_test_set[col]-X_train_set[col].mean())/X_train_set[col].std()
    X_train_set[col]=(X_train_set[col]-X_train_set[col].mean())/X_train_set[col].std()

X_train_set_np=np.array(X_train_set)
X_test_set_np=np.array(X_test_set)
Y_train_set_np=np.array(Y_train_set)
Y_test_set_np=np.array(Y_test_set)
skf=StratifiedKFold(n_splits=5, random_state=69)

n=[]

def score_model_RFE(model, params, cv=skf, features_to_select= n):
    smoter=SMOTE(random_state=42)

    scores=[]
    
    for train_fold_index, val_fold_index in cv.split(X_train_set_np, Y_train_set_np):
        X_train_fold, y_train_fold=X_train_set_np[train_fold_index], Y_train_set_np[train_fold_index]
        X_val_fold, y_val_fold=X_train_set_np[val_fold_index], Y_train_set_np[val_fold_index]
        X_train_fold_upsample, y_train_fold_upsample=smoter.fit_resample(X_train_fold, y_train_fold)
        RF=RFE(model(**params, n_jobs=-1),n_features_to_select= features_to_select,step=1, verbose=3).fit(X_train_fold_upsample, y_train_fold_upsample)
        score=roc_auc_score(y_val_fold, RF.predict(X_val_fold))
        scores.append(score)
        
    return np.array(scores)

max_depth=[1]
min_samples_split=[0.6, 2, 4, 5, 10, 20, 30]
min_samples_leaf=[1,5,20,25,30,40,50,60]
RFE_features_to_select= list(range(1,396))
n_estimators=500

def RFE_model(n):
    score_tracker=[]
    for c in min_samples_leaf:
        for b in min_samples_split:
            for a in max_depth:
                print("Features", n)
                print("Min samples", c)
                print("min samples split", b)
                print("max depth", a)
                example_params={'n_estimators': n_estimators, 'max_depth': a, 'min_samples_split': b, 'min_samples_leaf': c}
                example_params['roc_auc']=score_model_RFE(RandomForestClassifier, example_params, cv=skf, features_to_select= n).mean()
                example_params['n_features_selected']=n
                score_tracker.append(example_params)
    return np.array(score_tracker)

if __name__=='__main__':
    p=Pool()
    result= p.map(RFE_model, RFE_features_to_select)
    LIST_OF_RESULTS=[]
    for i in range(0,len(result)):
        for t in (0,(result[i].size-1)):
            LIST_OF_RESULTS.append(result[i][t])
            
            


