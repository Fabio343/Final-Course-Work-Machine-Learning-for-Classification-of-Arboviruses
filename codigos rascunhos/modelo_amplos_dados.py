# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:50:48 2020

@author: fabio343
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:51:19 2020

@author: fabio343
"""

#from __future__ import division
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import pickle
import sqlalchemy
from pandas.io import sql
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('figure', figsize=(11,9))
pd.options.display.html.table_schema = True
import datetime
from tqdm import tqdm
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import category_encoders as ce   # version 1.2.8
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from ipywidgets import interactive
from IPython.display import SVG,display
from graphviz import Source
import pydot
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot
import graphviz
from sklearn import tree
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer



def category(df_train,df_test,C):
    df_train_temp=df_train.copy()
    df_test_temp=df_test.copy()

    for col in C.columns:
        ordenad_labels=df_train.groupby([col])['fl_severidade'].mean().to_dict()
        #ordenal_labels={k:i for i,k in enumerate(ordenad_labels,0)}

        df_train_temp[col]=df_train[col].map(ordenad_labels)
        df_test_temp[col]=df_test[col].map(ordenad_labels)
 
    df_train_temp.drop(['fl_severidade'],axis=1,inplace=True)
    df_test_temp.drop(['fl_severidade'],axis=1,inplace=True)

    return df_train_temp,df_test_temp 

def category2(amostra_paci,C):
    df_train_temp=amostra_paci.copy()
    for col in C.columns:
        ordenad_labels=amostra_paci.groupby([col])['fl_severidade'].mean().to_dict()
        df_train_temp[col]=amostra_paci[col].map(ordenad_labels)
    return df_train_temp

def plot_roc_curve(y_true, y_score,nome,figsize=(10,6)):
    fpr, tpr,_ = roc_curve(y_true, y_score)
    plt.figure(figsize=figsize)
    auc_value = roc_auc_score(y_true, y_score)
    fig = plt.figure(figsize=(12,12))
    plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    fig.savefig("roc_curve_"+nome+".png")
    #plt.show()
    
def var_drop(df,colum):
  colunas=[]
  for k in colum:
     if df[k].sum()<=18:
       colunas.append(k)
  return colunas  

def logistic(x_train,x_test,y_train,y_test,X,fl,amostra_paci2,fl_a2,nome):
    logistic=LogisticRegressionCV(cv=5, random_state=44)
    logistic.fit(x_train,y_train)
    pred=logistic.predict_proba(x_train)
    
    coefs = np.abs(logistic.coef_[0])
    indices = np.argsort(coefs)[::-1]
    for feature,importancia in zip(x_train.columns,coefs[indices[:]]):
      print("{}:{}".format(feature, importancia))
    plt.figure(figsize=(15,15))
    plt.title("Feature importances (Logistica)")
    plt.bar(range(len(x_train.columns)),coefs[indices[:]],
       color="r", align="center")
    plt.xticks(range(len(x_train.columns)),x_train.columns[indices[:]], rotation=50, ha='right')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("logistica_feature_importance.png")
    plt.show()
    print(coefs[indices[:]],x_train.columns[indices[:]])
    amostra_=logistic.predict_proba(amostra_paci2)
    amostra_2=logistic.predict(amostra_paci2)
    amostra_paci2['result']=0
    amostra_paci2['probls']=0
    amostra_paci2['probls']=amostra_
    amostra_paci2['result']=amostra_2
    amostra_paci2['fl_severidade']=fl_a2
    amostra_paci2.to_csv('modelo_logistic.csv')
    print('Treinamento AUC-ROC:{}'.format(roc_auc_score(y_train,pred[:,1])))
    pred_2=logistic.predict_proba(x_test)
    print('Validacao AUC-ROC:{}'.format(roc_auc_score(y_test,pred_2[:,1])))
    yhat = logistic.predict_proba(x_train)
    yhat = yhat[:, 1] 
    print(pd.crosstab(y_train, logistic.predict(x_train)))
    print(classification_report(y_train, logistic.predict(x_train)))
    print('AUC: %0.2f' % roc_auc_score(y_train,yhat))
    plot_roc_curve(y_train,yhat,'logist_train')   
    yhat = logistic.predict_proba(x_test)
    yhat = yhat[:, 1] 
    print(pd.crosstab(y_test, logistic.predict(x_test)))
    print(classification_report(y_test, logistic.predict(x_test)))
    print('AUC: %0.2f' % roc_auc_score(y_test,yhat))
    plot_roc_curve(y_test,yhat,'logist_test')
    
def logistic_1(x_test,y_test,x_train_1,y_train_1,X,fl,amostra_paci2,fl_a2,nome):    
    print('verificação com down em treino')
    print()
    logistic=LogisticRegressionCV(cv=5, random_state=44)
    logistic.fit(x_train_1,y_train_1)
    pred=logistic.predict_proba(x_train_1)
    coefs = np.abs(logistic.coef_[0])
    indices = np.argsort(coefs)[::-1]
    for feature,importancia in zip(x_train_1.columns,coefs[indices[:]]):
      print("{}:{}".format(feature, importancia))
    plt.figure(figsize=(15,15))
    plt.title("Feature importances (Logistica)")
    plt.bar(range(len(x_train_1.columns)),coefs[indices[:]],
       color="r", align="center")
    plt.xticks(range(len(x_train_1.columns)),x_train_1.columns[indices[:]], rotation=50, ha='right')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("logistica_feature_importance_1.png")
    plt.show()
    print(coefs[indices[:]],x_train_1.columns[indices[:]])
    amostra_=logistic.predict_proba(amostra_paci2)
    amostra_2=logistic.predict(amostra_paci2)
    amostra_paci2['result']=0
    amostra_paci2['probls']=0
    amostra_paci2['probls']=amostra_
    amostra_paci2['result']=amostra_2
    amostra_paci2['fl_severidade']=fl_a2
    amostra_paci2.to_csv('modelo_logistic_1.csv')
    print('Treinamento AUC-ROC:{}'.format(roc_auc_score(y_train_1,pred[:,1])))
    pred_2=logistic.predict_proba(x_test)
    print('Validacao AUC-ROC:{}'.format(roc_auc_score(y_test,pred_2[:,1])))    
    yhat = logistic.predict_proba(x_train_1)
    yhat = yhat[:, 1] 
    print(pd.crosstab(y_train_1, logistic.predict(x_train_1)))
    print(classification_report(y_train_1, logistic.predict(x_train_1)))
    print('AUC: %0.2f' % roc_auc_score(y_train_1,yhat))
    plot_roc_curve(y_train_1,yhat,'logist_train_1')
    yhat = logistic.predict_proba(x_test)
    yhat = yhat[:, 1] 
    print(pd.crosstab(y_test, logistic.predict(x_test)))
    print(classification_report(y_test, logistic.predict(x_test)))
    print('AUC: %0.2f' % roc_auc_score(y_test,yhat))
    plot_roc_curve(y_test,yhat,'logist_test_1')

def complement_bayes(x_train,x_test,y_train,y_test,x_train_1,y_train_1,X,fl,amostra_paci3,fl_a3,nome):
    amostra_paci3_n=amostra_paci3.copy()
    x_train = preprocessing.normalize(x_train)
    x_train_1 = preprocessing.normalize(x_train_1)
    x_test = preprocessing.normalize(x_test)

    Complement=ComplementNB()
    Complement.fit(x_train,y_train)
    print(Complement.fit(x_train, y_train).feature_log_prob_)
    pred=Complement.predict_proba(x_train)
    amostra_=Complement.predict_proba(amostra_paci3)
    amostra_2=Complement.predict(amostra_paci3)
    amostra_paci3['result']=0
    amostra_paci3['probls']=0
    amostra_paci3['probls']=amostra_
    amostra_paci3['result']=amostra_2
    amostra_paci3['fl_severidade']=fl_a3
    amostra_paci3.to_csv('modelo_complement_bayes.csv')
    print('Treinamento AUC-ROC:{}'.format(roc_auc_score(y_train,pred[:,1])))
    pred_2=Complement.predict_proba(x_test)
    print('Validacao AUC-ROC:{}'.format(roc_auc_score(y_test,pred_2[:,1])))
    yhat = Complement.predict_proba(x_train)
    yhat = yhat[:, 1] 
    print(pd.crosstab(y_train, Complement.predict(x_train)))
    print(classification_report(y_train, Complement.predict(x_train)))
    print('AUC: %0.2f' % roc_auc_score(y_train,yhat))
    plot_roc_curve(y_train,yhat,'naive_train')
    yhat = Complement.predict_proba(x_test)
    yhat = yhat[:, 1] 
    print(pd.crosstab(y_test, Complement.predict(x_test)))
    print(classification_report(y_test, Complement.predict(x_test)))
    print('AUC: %0.2f' % roc_auc_score(y_test,yhat))
    plot_roc_curve(y_test,yhat,'naive_test')
        
    print('verificação com down em treino')
    print()
    Complement=ComplementNB()
    Complement.fit(x_train_1,y_train_1)
    print(Complement.fit(x_train_1, y_train_1).feature_log_prob_)
    pred=Complement.predict_proba(x_train_1)
    amostra_=Complement.predict_proba(amostra_paci3_n)
    amostra_2=Complement.predict(amostra_paci3_n)
    amostra_paci3_n['result']=0
    amostra_paci3_n['probls']=0
    amostra_paci3_n['probls']=amostra_
    amostra_paci3_n['result']=amostra_2
    amostra_paci3_n['fl_severidade']=fl_a3
    amostra_paci3_n.to_csv('modelo_complement_bayes_1.csv')
    print('Treinamento AUC-ROC:{}'.format(roc_auc_score(y_train_1,pred[:,1])))
    pred_2=Complement.predict_proba(x_test)
    print('Validacao AUC-ROC:{}'.format(roc_auc_score(y_test,pred_2[:,1])))
    yhat = Complement.predict_proba(x_train_1)
    yhat = yhat[:, 1] 
    print(pd.crosstab(y_train_1, Complement.predict(x_train_1)))
    print(classification_report(y_train_1, Complement.predict(x_train_1)))
    print('AUC: %0.2f' % roc_auc_score(y_train_1,yhat))
    plot_roc_curve(y_train_1,yhat,'naive_train_1')
    yhat = Complement.predict_proba(x_test)
    yhat = yhat[:, 1] 
    print(pd.crosstab(y_test, Complement.predict(x_test)))
    print(classification_report(y_test, Complement.predict(x_test)))
    print('AUC: %0.2f' % roc_auc_score(y_test,yhat))
    plot_roc_curve(y_test,yhat,'naive_test_1')

def arvore_dec(x_train,x_test,y_train,y_test,X,fl,amostra_paci4,fl_a4,nome):
    parameters = {'max_depth':range(3,20)}
    arvore=GridSearchCV(DecisionTreeClassifier(),parameters, n_jobs=4)
    arvore_treinada = arvore.fit(x_train,y_train)
    AAA=list(arvore_treinada.best_params_.values())
    print(AAA)
    print()
    print(AAA[-1])
    arvore=DecisionTreeClassifier(max_depth=(14))
    arvore_treinada = arvore.fit(x_train,y_train)
    coefs = arvore_treinada.feature_importances_
    indices = np.argsort(coefs)[::-1]
    for feature,importancia in zip(amostra_paci4.columns,arvore_treinada.feature_importances_):
      print("{}:{}".format(feature, importancia))
        
    plt.figure(figsize=(15,15))
    plt.title("Feature importances (Arvore)")
    plt.bar(range(len(x_train.columns)), coefs[indices[:]],
       color="r", align="center")
    plt.xticks(range(len(x_train.columns)),x_train.columns[indices[:]], rotation=50, ha='right')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("arvore_feature_importance.png")
    plt.show()
    print(coefs[indices[:]],x_train.columns[indices[:]])
    resultado= arvore_treinada.predict(x_test)
    print(pd.crosstab(y_train, arvore_treinada.predict(x_train)))
    print(metrics.classification_report(y_train,arvore_treinada.predict(x_train)))
    yhat = arvore_treinada.predict_proba(x_train)
    yhat = yhat[:, 1]
    plot_roc_curve(y_train,yhat,'avore_train')
    print(pd.crosstab(y_test, arvore_treinada.predict(x_test)))
    print(metrics.classification_report(y_test,arvore_treinada.predict(x_test)))
    yhat = arvore_treinada.predict_proba(x_test)
    yhat = yhat[:, 1]
    plot_roc_curve(y_test,yhat,'avore_test')
    amostra_=arvore_treinada.predict_proba(amostra_paci4)
    amostra_2=arvore_treinada.predict(amostra_paci4)
    amostra_paci4['result']=0
    amostra_paci4['probls']=0
    amostra_paci4['probls']=amostra_
    amostra_paci4['result']=amostra_2
    amostra_paci4['fl_severidade']=fl_a4
    amostra_paci4.to_csv('modelo_arvore.csv')
    fig = plt.figure(figsize=(25,25))
    tree.plot_tree(arvore_treinada, 
                   feature_names=x_train.columns,  
                   class_names='fl_severidade',
                   filled=True)
    fig.savefig("decistion_tree.png")
 
def arvore_dec1(x_test,y_test,x_train_1,y_train_1,X,fl,amostra_paci4,fl_a4,nome):    
    print('verificação com down em treino')
    print()
    parameters = {'max_depth':range(3,20)}
    arvore=GridSearchCV(DecisionTreeClassifier(),parameters, n_jobs=4)
    arvore_treinada = arvore.fit(x_train_1,y_train_1)
    AAA=list(arvore_treinada.best_params_.values())
    print(AAA)
    print()
    print(AAA[-1])
    arvore=DecisionTreeClassifier(max_depth=(14))
    arvore_treinada = arvore.fit(x_train_1,y_train_1)
    coefs = arvore_treinada.feature_importances_
    indices = np.argsort(coefs)[::-1]
    for feature,importancia in zip(amostra_paci4.columns,arvore_treinada.feature_importances_):
      print("{}:{}".format(feature, importancia))
        
    plt.figure(figsize=(15,15))
    plt.title("Feature importances (Arvore)")
    plt.bar(range(len(x_train_1.columns)), coefs[indices[:]],
       color="r", align="center")
    plt.xticks(range(len(x_train_1.columns)),x_train_1.columns[indices[:]], rotation=50, ha='right')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("arvore_feature_importance_1.png")
    plt.show()
    print(coefs[indices[:]],x_train_1.columns[indices[:]])
    resultado= arvore_treinada.predict(x_test)
    print(pd.crosstab(y_train_1, arvore_treinada.predict(x_train_1)))
    print(metrics.classification_report(y_train_1,arvore_treinada.predict(x_train_1)))
    yhat = arvore_treinada.predict_proba(x_train_1)
    yhat = yhat[:, 1]
    plot_roc_curve(y_train_1,yhat,'avore_train_1')
    print(pd.crosstab(y_test, arvore_treinada.predict(x_test)))
    print(metrics.classification_report(y_test,arvore_treinada.predict(x_test)))
    yhat = arvore_treinada.predict_proba(x_test)
    yhat = yhat[:, 1]
    plot_roc_curve(y_test,yhat,'avore_test_1')
    amostra_=arvore_treinada.predict_proba(amostra_paci4)
    amostra_2=arvore_treinada.predict(amostra_paci4)
    amostra_paci4['result']=0
    amostra_paci4['probls']=0
    amostra_paci4['probls']=amostra_
    amostra_paci4['result']=amostra_2
    amostra_paci4['fl_severidade']=fl_a4
    amostra_paci4.to_csv('modelo_arvore_1.csv')
    fig = plt.figure(figsize=(25,25))
    tree.plot_tree(arvore_treinada, 
                   feature_names=x_train_1.columns,  
                   class_names='fl_severidade',
                   filled=True)
    fig.savefig("decistion_tree_1.png")

def random_f(x_train,x_test,y_train,y_test,X,fl,amostra_pac5,fl_a5,nome):
    tuned_parameters = [{'n_estimators': [100,125,150,175,200,225,250,300],'max_depth': [4,5,6,7,8],'n_jobs':[3,4,5,6]}]
    arvore= GridSearchCV(RandomForestClassifier(), tuned_parameters, cv= 4, scoring='roc_auc')
    print('Melhores parametros')
    arvore_treinada = arvore.fit(x_train,y_train)
    AAA=list(arvore_treinada.best_params_.values())
    print(AAA)
    print()
    print(AAA[:])
    arvore=RandomForestClassifier(max_depth=AAA[0],n_estimators=AAA[1])
    arvore_treinada = arvore.fit(x_train,y_train)
    coefs = arvore_treinada.feature_importances_
    indices = np.argsort(coefs)[::-1]
    for feature,importancia in zip(amostra_paci5.columns,arvore_treinada.feature_importances_):
      print("{}:{}".format(feature, importancia))
        
    plt.figure(figsize=(15,15))
    plt.title("Feature importances (Random)")
    plt.bar(range(len(x_train.columns)), coefs[indices[:]],
       color="r", align="center")
    plt.xticks(range(len(x_train.columns)),x_train.columns[indices[:]], rotation=50, ha='right')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("random_feature_importance.png")
    plt.show()
    print(coefs[indices[:]],x_train.columns[indices[:]])
    resultado= arvore_treinada.predict(x_test)
    print(pd.crosstab(y_train, arvore_treinada.predict(x_train)))
    print(metrics.classification_report(y_train,arvore_treinada.predict(x_train))) 
    yhat = arvore_treinada.predict_proba(x_train)
    yhat = yhat[:, 1]
    plot_roc_curve(y_train,yhat,'random_train')
    print(pd.crosstab(y_test, arvore_treinada.predict(x_test)))
    print(metrics.classification_report(y_test,arvore_treinada.predict(x_test)))
    yhat = arvore_treinada.predict_proba(x_test)
    yhat = yhat[:, 1]
    plot_roc_curve(y_test,yhat,'random_test')
    amostra_=arvore_treinada.predict_proba(amostra_paci5)
    amostra_2=arvore_treinada.predict(amostra_paci5)
    amostra_paci5['result']=0
    amostra_paci5['probls']=0
    amostra_paci5['probls']=amostra_
    amostra_paci5['result']=amostra_2
    amostra_paci5['fl_severidade']=fl_a5
    amostra_paci5.to_csv('modelo_random.csv')
    
def random_f1(x_test,y_test,x_train_1,y_train_1,X,fl,amostra_paci5,fl_a5,nome):    
    print('verificação com down em treino')
    print()
    tuned_parameters = [{'n_estimators': [100,125,150,175,200,225,250,300],'max_depth': [4,5,6,7,8],'n_jobs':[3,4,5,6]}]
    arvore= GridSearchCV(RandomForestClassifier(), tuned_parameters, cv= 4, scoring='roc_auc')
    print('Melhores parametros')
    arvore_treinada = arvore.fit(x_train_1,y_train_1)
    AAA=list(arvore_treinada.best_params_.values())
    print(AAA)
    print()
    print(AAA[:])
    arvore=RandomForestClassifier(max_depth=AAA[0],n_estimators=AAA[1])
    arvore_treinada = arvore.fit(x_train_1,y_train_1)
    coefs = arvore_treinada.feature_importances_
    indices = np.argsort(coefs)[::-1]
    for feature,importancia in zip(amostra_paci5.columns,arvore_treinada.feature_importances_):
      print("{}:{}".format(feature, importancia))
        
    plt.figure(figsize=(15,15))
    plt.title("Feature importances (Random)")
    plt.bar(range(len(x_train_1.columns)), coefs[indices[:]],
       color="r", align="center")
    plt.xticks(range(len(x_train_1.columns)),x_train_1.columns[indices[:]], rotation=50, ha='right')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("random_feature_importance_1.png")
    plt.show()
    print(coefs[indices[:]],x_train_1.columns[indices[:]])
    resultado= arvore_treinada.predict(x_test)
    print(pd.crosstab(y_train_1, arvore_treinada.predict(x_train_1)))
    print(metrics.classification_report(y_train_1,arvore_treinada.predict(x_train_1))) 
    yhat = arvore_treinada.predict_proba(x_train_1)
    yhat = yhat[:, 1]
    plot_roc_curve(y_train_1,yhat,'random_train_1')
    print(pd.crosstab(y_test, arvore_treinada.predict(x_test)))
    print(metrics.classification_report(y_test,arvore_treinada.predict(x_test)))
    yhat = arvore_treinada.predict_proba(x_test)
    yhat = yhat[:, 1]
    plot_roc_curve(y_test,yhat,'random_test_1')
    amostra_=arvore_treinada.predict_proba(amostra_paci5)
    amostra_2=arvore_treinada.predict(amostra_paci5)
    amostra_paci5['result']=0
    amostra_paci5['probls']=0
    amostra_paci5['probls']=amostra_
    amostra_paci5['result']=amostra_2
    amostra_paci5['fl_severidade']=fl_a5
    amostra_paci5.to_csv('modelo_random_1.csv')


base_unificada5=pd.read_excel('var_nov_tcc.xlsx',sep=';',error_bad_lines=False)
base_=pd.read_excel('Levantamento_clinico_PALMAS_ARBOBIOS_2019.xlsx',sep=';',error_bad_lines=False)
dados_analises6=pd.read_csv('arbobios_erika_dez2019.csv',sep=';',error_bad_lines=False)
print()
for k in base_.columns:
    if k not in dados_analises6.columns:
        base_.drop(k, axis=1, inplace=True)

base_total=pd.concat([dados_analises6, base_], ignore_index=True)
base_unificada5=base_unificada5.merge(base_total[['Subject_ID','CLASSIFICACAO FINAL']],how='left',on=['Subject_ID'])


ids=base_total[['Subject_ID']]
col=[]
col.append('Subject_ID')

#Verificando o tipo de dado presente e quais seriam as colunas.
A=base_unificada5.select_dtypes(include=['int64'])
B=base_unificada5.select_dtypes(include=['float64'])
B.to_csv('floats.csv')
print()
C=base_unificada5.select_dtypes(include=['object'])
C.to_csv('strings.csv')

base_unificada5.to_csv('variavesi.csv')
# lista de variaveis para serem removidas do 
#processo por serem basicamente informações
#de telefone,contato e datas 


base_unificada5.to_csv('base1.csv')

variaveis_dummy_2=[]
for k in ['febre_7dias','sangramento','raca','ja_teve_dengue','ja_teve_chikungunya','ja_teve_zika','tem_alguma_doenca_das_articulacoes',	
'exantema','edema','sinais_artrite','outros_sinais_alarme','esteve_hospitalizado_D07','esteve_hospitalizado_agravamento_dengue_D07','esteve_hospitalizado_D14',	
'esteve_hospitalizado_agravamento_dengue_D14']:
  base_dummy_novas= pd.get_dummies(base_unificada5[k],prefix=k)
  base_unificada5=base_unificada5.merge(base_dummy_novas,left_index=True,right_index=True)
 
  for i in base_dummy_novas.columns:
    variaveis_dummy_2.append(i)
    

for k in ['febre_7dias','sangramento','raca','ja_teve_dengue','ja_teve_chikungunya','ja_teve_zika','tem_alguma_doenca_das_articulacoes',	
'exantema','edema','sinais_artrite','outros_sinais_alarme','esteve_hospitalizado_D07','esteve_hospitalizado_agravamento_dengue_D07','esteve_hospitalizado_D14',	
'esteve_hospitalizado_agravamento_dengue_D14']:
  base_unificada5.drop(k, axis=1, inplace=True)

base_unificada5.reset_index()
base_unificada5.to_csv('base2.csv')

base_unificada5['fl_severidade'] = 0
base_unificada5['age_crianca'] = 0
base_unificada5['age_idoso'] = 0
base_unificada5['fl_sexo'] = 0

base_unificada5['fl_severidade'][(base_unificada5['CLASSIFICACAO FINAL']==3.0)] = 1
base_unificada5['age_crianca'][(base_unificada5['age']<=11)] = 1
base_unificada5['age_adolecente']=[1 if 11<x<=20  else 0 for x in base_unificada5['age']]
base_unificada5['age_adulto']=[1 if 20<x<60  else 0 for x in base_unificada5['age']]
base_unificada5['age_idoso']=[1 if x>=60  else 0 for x in base_unificada5['age']]
base_unificada5['fl_sexo'][(base_unificada5['sexo']=='feminino')] = 1
base_unificada5_filtrada=base_unificada5

base_unificada5_filtrada.drop('sexo', axis=1, inplace=True)
base_unificada5_filtrada.drop('CLASSIFICACAO FINAL', axis=1, inplace=True)
base_unificada5_filtrada.drop('age', axis=1, inplace=True)

C=base_unificada5_filtrada.select_dtypes(include=['object'])
base_unificada5_filtrada.to_csv('base3.csv')


corelacao=base_unificada5_filtrada.corr(method ='spearman')['fl_severidade']
corelacao.to_csv('correlation_dados_tarq.csv')

lista_varia=var_drop(base_unificada5_filtrada,variaveis_dummy_2)
len(lista_varia)
for k in lista_varia:
    base_unificada5_filtrada.drop(k,axis=1,inplace=True)

C=base_unificada5_filtrada.select_dtypes(include=['object'])
C.to_csv('strings.csv')
C.columns 

base_unificada5_filtrada[C.columns]=base_unificada5_filtrada[C.columns].fillna('Miss')
base_unificada5_filtrada=base_unificada5_filtrada.fillna(0.0)   
base_unificada5_filtrada.to_csv('base_total.csv')

base_para_score=base_unificada5_filtrada.copy()
amostra_paci=base_unificada5_filtrada[base_unificada5_filtrada['Subject_ID'].isin(ids['Subject_ID'])].reset_index()
from sklearn.utils import resample
'''amostra_paci_zero=amostra_paci[amostra_paci['fl_severidade']==0]
amostra_paci_um=amostra_paci[amostra_paci['fl_severidade']==1]
amostra_paci_um_up=resample(amostra_paci_um,
                         replace=True,
                         n_samples=len(amostra_paci_zero),
                         random_state=123)

amostra_paci=pd.concat([amostra_paci_zero,amostra_paci_um_up])
print(amostra_paci[amostra_paci['fl_severidade']==0].count())
'''
amostra_paci_21=amostra_paci.copy()
amostra_paci_31=amostra_paci.copy()
amostra_paci.drop('Subject_ID', axis=1, inplace=True)
amostra_paci.drop('index', axis=1, inplace=True)

'''
for k in amostra_paci.columns:
   if k not in ['SAQ_PressÃ£o baixa, escurecimento da vista, sudorese ou desmaio ao se levantar',
       'esteve_hospitalizado_D07_NÃ£o',
       'tem_alguma_doenca_das_articulacoes_NÃ£o',
       'SA_D07_PressÃ£o baixa, escurecimento da vista, sudorese ou desmaio ao se levantar',
       'SAQ_Muito sono, ideias confusas, fala atrapalhada',
       'SA_D07_Sangramento na urina ou fezes', 'fl_sexo',
       'sintoma_febre_7dias_Dor atrÃ¡s dos olhos',
       'outros_sinais_alarme_Sem registro',
       'esteve_hospitalizado_agravamento_dengue_D14_NÃ£o sabe',
       'sintoma_febre_7dias_Dor de cabeÃ§a',
       'SA_D07_Dor intensa e contÃ­nua no abdome (barriga), espontÃ¢nea ou ao apertar',
       'sintoma_febre_7dias_Dor muscular (costas, coxas, panturrilhas, braÃ§os)',
       'SAQ_Queda abrupta de plaquetas',
       'SA_D14_Dor intensa e contÃ­nua no abdome (barriga), espontÃ¢nea ou ao apertar',
       'age_crianca', 'SAQ_VÃ´mitos persistentes',
       'ja_teve_zika_NÃ£o que eu saiba', 'sangramento_Sim', 'exantema_NÃ£o',
       'sintoma_febre_7dias_Dor nas articulaÃ§Ãµes (juntas)',
       "SA_D07_! 'Muito sono, ideias confusas, fala atrapalhada '",
       'age_adulto', 'raca_Branca', 'age_idoso', 'outros_sinais_alarme_NÃ£o',
       'SA_D07_Manchas vermelhas na pele', 'SA_D14_Manchas vermelhas na pele',
       'esteve_hospitalizado_agravamento_dengue_D07_NÃ£o sabe', 'raca_Parda',
       'ja_teve_dengue_NÃ£o que eu saiba',
       'SA_D14_PressÃ£o baixa, escurecimento da vista, sudorese ou desmaio ao se levantar',
       'sangramento_NÃ£o', 'SA_D14_Sem sinais',
       'ja_teve_dengue_Sim, por diagnÃ³stico do mÃ©dico',
       "SA_D07_! 'Sangramento de mucosas (nariz, gengiva, etc) '",
       'SAQ_Sangramento de mucosas (nariz, gengiva, etc)',
       'SA_D07_VÃ´mitos persistentes', 'esteve_hospitalizado_D14_NÃ£o',
       'SA_D07_VÃ´mitos com sangue',
       'SAQ_Dor intensa e contÃ­nua no abdome (barriga), espontÃ¢nea ou ao apertar',
       'sintoma_febre_7dias_Manchas na pele','fl_severidade']:
       amostra_paci.drop(k, axis=1, inplace=True)
'''
amostra_paci=category2(amostra_paci,C)
len(amostra_paci.columns)
amostra_paci=amostra_paci.fillna(0.0) 

variaveis_features=[]
for k in amostra_paci.columns:
 if k!='fl_severidade':
     variaveis_features.append(k)

X = amostra_paci[variaveis_features].values
Y = amostra_paci['fl_severidade'].values
model = LogisticRegression()
rfe = RFE(model,len(variaveis_features))
fit = rfe.fit(X, Y)
print("Numero de  Features: %d" % fit.n_features_)
print("Features selecionadas: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print('')

len(fit.ranking_)
len(variaveis_features)
drops_feature=[]
for k in range(len(fit.ranking_)):
   if  fit.ranking_[k] !=1:
       drops_feature.append(variaveis_features[k])
len(drops_feature) 
for k in variaveis_features:
    if k not in drops_feature:
        print(k)
        
for L in drops_feature: 
  amostra_paci.drop(L, axis=1, inplace=True)
       
amos=amostra_paci['fl_severidade'].sum()
print(amos)

x_train,x_test,y_train,y_test=train_test_split(amostra_paci,amostra_paci['fl_severidade'],test_size=0.3,random_state=0)
x_train_1=x_train.copy()
y_train_1=y_train.copy()
x_train.columns

x_train['fl_severidade']=y_train
x_train_zero=x_train[x_train['fl_severidade']==0]
x_train_um=x_train[x_train['fl_severidade']==1]
x_train_um_up=resample(x_train_um,
                         replace=True,
                         n_samples=len(x_train_zero),
                         random_state=123)
x_train=pd.concat([x_train_zero,x_train_um_up])
print(x_train[x_train['fl_severidade']==0].count())
y_train=x_train['fl_severidade']
x_train.drop('fl_severidade',axis=1)


x_train_1['fl_severidade']=y_train_1
x_train_zero=x_train_1[x_train_1['fl_severidade']==0]
x_train_um=x_train_1[x_train_1['fl_severidade']==1]
x_train_zero_down=resample(x_train_zero,
                         replace=True,
                         n_samples=len(x_train_um),
                         random_state=123)
x_train_1=pd.concat([x_train_zero_down,x_train_um])
print(x_train_1[x_train_1['fl_severidade']==0].count())
y_train_1=x_train_1['fl_severidade']
x_train_1.drop('fl_severidade',axis=1)

#################################################################
#LIMPEZA DE VARIAVEIS CONSTANTES
##################################################################
vl_limpa_const=0.1 # variaveis com 99% dos campos repetidos
limpa_const = VarianceThreshold(threshold= vl_limpa_const)
limpa_const.fit(amostra_paci)
vars_const = [v for v in amostra_paci.columns if v not in  amostra_paci.columns[limpa_const.get_support()]]
qt_var = len([v for v in amostra_paci.columns if v not in  amostra_paci.columns[limpa_const.get_support()]])
 
print('Existem {} variaveis constantes com limite de {}'.format(qt_var,vl_limpa_const))
print('Variaveis constantes com limite de {}'.format(vl_limpa_const))
print(vars_const)
d = {'vars_const': vars_const}
df = pd.DataFrame(data=d)
df.to_csv('vars_const.csv') 
print('')
pickle_lista = open("vars_const.pickle","wb")
pickle.dump(vars_const, pickle_lista)
pickle_lista.close()

amostra_paci2=amostra_paci.copy()
amostra_paci3=amostra_paci.copy()
amostra_paci4=amostra_paci.copy()
amostra_paci5=amostra_paci.copy()

fl_a1=amostra_paci[['fl_severidade']]
fl_a2=amostra_paci2[['fl_severidade']]
fl_a3=amostra_paci3[['fl_severidade']]
fl_a4=amostra_paci4[['fl_severidade']]
fl_a5=amostra_paci5[['fl_severidade']]

corelacao=amostra_paci.corr(method ='spearman')
from sklearn.metrics import roc_auc_score
corelacao.to_csv('correlation_dados.csv')
x_train=x_train.fillna(0) 
x_train_1=x_train_1.fillna(0) 
x_test=x_test.fillna(0)
x_train.drop(['fl_severidade'],axis=1,inplace=True)
x_train_1.drop(['fl_severidade'],axis=1,inplace=True)
x_test.drop(['fl_severidade'],axis=1,inplace=True)
fl=amostra_paci['fl_severidade']
amostra_paci.drop(['fl_severidade'],axis=1,inplace=True) 
amostra_paci2.drop(['fl_severidade'],axis=1,inplace=True) 
amostra_paci3.drop(['fl_severidade'],axis=1,inplace=True) 
amostra_paci4.drop(['fl_severidade'],axis=1,inplace=True)
amostra_paci5.drop(['fl_severidade'],axis=1,inplace=True) 
y_train=y_train.fillna(0)
y_train_1=y_train_1.fillna(0) 
y_test=y_test.fillna(0)

amostra_paci2a=amostra_paci2.copy()
amostra_paci3a=amostra_paci3.copy()
amostra_paci4a=amostra_paci4.copy()
amostra_paci5a=amostra_paci5.copy()
print('Modelo Logistica')
print(logistic(x_train,x_test,y_train,y_test,amostra_paci,fl,amostra_paci2,fl_a2,nome='log'))
print('')
print('')
print('Modelo Logistica Down')
print(logistic_1(x_test,y_test,x_train_1,y_train_1,amostra_paci,fl,amostra_paci2a,fl_a2,nome='log'))
print('')
print('')
print('Modelo Bayes Complement')
print(complement_bayes(x_train,x_test,y_train,y_test,x_train_1,y_train_1,amostra_paci,fl,amostra_paci3,fl_a3,nome='bayes'))
print('')
print('')
print('Modelo arvore')
print(arvore_dec(x_train,x_test,y_train,y_test,amostra_paci,fl,amostra_paci4,fl_a4,nome='arvor'))
print('')
print('Modelo arvore Down')
print(arvore_dec1(x_test,y_test,x_train_1,y_train_1,amostra_paci,fl,amostra_paci4a,fl_a4,nome='arvor'))
print('')
print('Modelo Random Forest')
print(random_f(x_train,x_test,y_train,y_test,amostra_paci,fl,amostra_paci5,fl_a5,nome='random'))
print('')
print('Modelo Random Forest Down')
print(random_f1(x_test,y_test,x_train_1,y_train_1,amostra_paci,fl,amostra_paci5a,fl_a5,nome='random'))
