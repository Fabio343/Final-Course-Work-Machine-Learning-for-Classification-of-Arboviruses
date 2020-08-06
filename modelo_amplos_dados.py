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
    logistic=LogisticRegression(random_state=44)
    logistic.fit(x_train,y_train)
    pred=logistic.predict_proba(x_train)
    for feature,importancia in zip(x_train.columns,logistic.coef_[0]):
      print("{}:{}".format(feature, importancia))
    amostra_=logistic.predict_proba(amostra_paci2)
    amostra_2=logistic.predict(amostra_paci2)
    amostra_paci2['result']=0
    amostra_paci2['probls']=0
    amostra_paci2['probls']=amostra_
    amostra_paci2['result']=amostra_2
    amostra_paci2['fl_severidade']=fl_a2
    amostra_paci2.to_csv('modelo_logistic_novo.csv')
    print('Treinamento AUC-ROC:{}'.format(roc_auc_score(y_train,pred[:,1])))
    pred_2=logistic.predict_proba(x_test)
    print('Validacao AUC-ROC:{}'.format(roc_auc_score(y_test,pred_2[:,1])))
    #print(logistic.coef_)
    #print(logistic.predict_proba(X))
    yhat = logistic.predict_proba(X)
    yhat = yhat[:, 1] 
    print(pd.crosstab(fl, logistic.predict(X)))
    print(classification_report(fl, logistic.predict(X)))
    print('AUC: %0.2f' % roc_auc_score(fl,yhat))
    plot_roc_curve(fl,yhat,nome)

def complement_bayes(x_train,x_test,y_train,y_test,X,fl,amostra_paci3,fl_a3,nome):
    Complement=ComplementNB()
    Complement.fit(x_train,y_train)
    for feature,importancia in zip(x_train.columns,Complement.feature_log_prob_):
      print("{}:{}".format(feature, importancia))
    pred=Complement.predict_proba(x_train)
    amostra_=Complement.predict_proba(amostra_paci3)
    amostra_2=Complement.predict(amostra_paci3)
    amostra_paci3['result']=0
    amostra_paci3['probls']=0
    amostra_paci3['probls']=amostra_
    amostra_paci3['result']=amostra_2
    amostra_paci3['fl_severidade']=fl_a3
    amostra_paci3.to_csv('modelo_complement_bayes_novo.csv')
    print('Treinamento AUC-ROC:{}'.format(roc_auc_score(y_train,pred[:,1])))
    pred_2=Complement.predict_proba(x_test)
    print('Validacao AUC-ROC:{}'.format(roc_auc_score(y_test,pred_2[:,1])))
    #print(Complement.predict_proba(X))
    yhat = Complement.predict_proba(X)
    yhat = yhat[:, 1] 
    print(pd.crosstab(fl, Complement.predict(X)))
    print(classification_report(fl, Complement.predict(X)))
    print('AUC: %0.2f' % roc_auc_score(fl,yhat))
    plot_roc_curve(fl,yhat,nome)

def arvore_dec(x_train,x_test,y_train,y_test,X,fl,amostra_paci4,fl_a4,nome):
    arvore = DecisionTreeClassifier()
   # Treinando o modelo de arvore de decisão:
    arvore_treinada = arvore.fit(x_train,y_train)
    for feature,importancia in zip(amostra_paci4.columns,arvore_treinada.feature_importances_):
      print("{}:{}".format(feature, importancia))
    resultado= arvore_treinada.predict(x_test)
    x_test['result'] = arvore_treinada.predict(x_test)
    x_test['fl']=y_test
    print(metrics.classification_report(y_test,resultado))
    #tree.plot_tree(arvore_treinada) 

    amostra_=arvore_treinada.predict_proba(amostra_paci4)
    amostra_2=arvore_treinada.predict(amostra_paci4)
    
    yhat = arvore_treinada.predict_proba(X)
    yhat = yhat[:, 1] 
    print(pd.crosstab(fl, arvore_treinada.predict(X)))
    print(classification_report(fl, arvore_treinada.predict(X)))
    print('AUC: %0.2f' % roc_auc_score(fl,yhat))
    plot_roc_curve(fl,yhat,nome)

    amostra_paci4['result']=0
    amostra_paci4['probls']=0
    amostra_paci4['probls']=amostra_
    amostra_paci4['result']=amostra_2
    amostra_paci4['fl_severidade']=fl_a4
    amostra_paci4.to_csv('modelo_arvore_novo.csv')

    fig = plt.figure(figsize=(45,40))
    tree.plot_tree(arvore_treinada, 
                   feature_names=x_train.columns,  
                   class_names='fl_severidade',
                   filled=True)
    fig.savefig("decistion_tree.png")
    
### Definição de paramentros iniciais para o tratamento de dados
### Selecionando as informações presentes em arquivos .csv
### De modo a cruzar com outros arquivos para unificar,
### Em um unico dataframe os dados facilitando na manipulação.

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

# processo que usamos para separar variaveis com mais de 1 informação
# dentro de uma mesma celula da base 
# com essa separação podemos criar flags para cada sintoma
# deixando mais claro o impacto de cada sintoma no 
# processo final 
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

#Marcando a flag para a modelagem considerando 1 todos os casos severos e 0
#casos não severos 

base_unificada5['fl_severidade'] = 0
base_unificada5['fl_severidade'][(base_unificada5['CLASSIFICACAO FINAL']==3.0)] = 1
base_unificada5['fl_sexo'] = 0
base_unificada5['fl_sexo'][(base_unificada5['sexo']=='feminino')] = 1
base_unificada5_filtrada=base_unificada5
#import datetime
#base_unificada5['DOB']=pd.to_datetime(base_unificada5['DOB'],errors='coerce').replace(np.nan,datetime.datetime.now())
#base_unificada5['febre_7dias_quando']=pd.to_datetime(base_unificada5['febre_7dias_quando'],errors='coerce').replace(np.nan,datetime.datetime.now())

#base_unificada5=base_unificada5.reset_index()
#base_unificada5['idade'] = 0
#for L in range(len(base_unificada5['DOB'])):
#     difff=base_unificada5['febre_7dias_quando'][L]-base_unificada5['DOB'][L]
#     base_unificada5['idade'][L]=int((difff.days + difff.seconds/86400)/365.2425)
 
#Filtrando apenas os pacientes com dengue as demais
#aborviroses serão trabalhadas em outra etapa do processo
#base_unificada5_filtrada=base_unificada5[base_unificada5['exam1_agent'] =='DENV'].reset_index()
#base_unificada5_filtrada=base_unificada5_filtrada[(base_unificada5_filtrada['exam1_resultado_desc'] =='Reactive') | (base_unificada5_filtrada['exam2_resultado_desc'] =='Reactive') | (base_unificada5_filtrada['exam3_resultado_desc'] =='Reactive') | (base_unificada5_filtrada['exam4_resultado_desc'] =='Reactive') | (base_unificada5_filtrada['exam5_resultado_desc'] =='Reactive') | (base_unificada5_filtrada['exam6_resultado_desc'] =='Reactive')].reset_index()

#Relizando a conversão de valores string para numericos
#Com o objetivo de simplificar a aplicação nas tecnicas de modelagem
#A alteração é dada com base na frequencia em relação com a resposta inicial desejada
#Ou seja inicialmente definimos pacientes com casos graves.
#Apos marcação realizamos um tipo de replace no qual 
#alteramos todos os campos com a nova marcação e salvamos os arquivos em formato pickle

base_unificada5_filtrada.drop('sexo', axis=1, inplace=True)
base_unificada5_filtrada.drop('CLASSIFICACAO FINAL', axis=1, inplace=True)
#base_unificada5_filtrada.drop('exam1_agent', axis=1, inplace=True)
#base_unificada5_filtrada.drop('exam1_resultado_desc', axis=1, inplace=True)
#base_unificada5_filtrada.drop('exam2_resultado_desc', axis=1, inplace=True)
#base_unificada5_filtrada.drop('exam3_resultado_desc', axis=1, inplace=True)
#base_unificada5_filtrada.drop('exam4_resultado_desc', axis=1, inplace=True)
#base_unificada5_filtrada.drop('exam5_resultado_desc', axis=1, inplace=True)
#base_unificada5_filtrada.drop('exam6_resultado_desc', axis=1, inplace=True)

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

#Selecionando amostra para modelagem após os tratamentos 
#Realizados em todos os dados
# E separando em treino e teste
base_para_score=base_unificada5_filtrada.copy()
#amostra_paci =base_unificada5_filtrada.copy() #.sample(frac=0.7, replace=False)
#base_unificada5_filtrada.drop('level_0', axis=1, inplace=True)

amostra_paci=base_unificada5_filtrada[base_unificada5_filtrada['Subject_ID'].isin(ids['Subject_ID'])].reset_index()

amostra_paci_21=amostra_paci.copy()
amostra_paci_31=amostra_paci.copy()
amostra_paci.drop('Subject_ID', axis=1, inplace=True)
#amostra_paci.drop('level_0', axis=1, inplace=True)
amostra_paci.drop('index', axis=1, inplace=True)

amostra_paci=category2(amostra_paci,C)
len(amostra_paci.columns)
amostra_paci=amostra_paci.fillna(0.0) 

variaveis_features=[]
for k in amostra_paci.columns:
 if k!='fl_severidade':
     variaveis_features.append(k)

X = amostra_paci[variaveis_features].values
Y = amostra_paci['fl_severidade'].values
# feature extraction
model = LogisticRegression()
#rfe = RFE(model, len(amostra_paci[variaveis_features].columns))
rfe = RFE(model,30)
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
  
#variaveis_features
#pca = PCA(n_components=30)
#fit = pca.fit(X)
# summarize components
#print("Explained Variance: %s" % fit.explained_variance_ratio_)
#print(fit.components_)
#print('')
x_train,x_test,y_train,y_test=train_test_split(amostra_paci,amostra_paci['fl_severidade'],test_size=0.5,random_state=0)
test=y_test.sum()
print(test)
#while test<40 or test>25:
#    x_train,x_test,y_train,y_test=train_test_split(amostra_paci,amostra_paci['fl_severidade'],test_size=0.5)
#    test=y_test.sum()

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

fl_a1=amostra_paci[['fl_severidade']]
fl_a2=amostra_paci2[['fl_severidade']]
fl_a3=amostra_paci3[['fl_severidade']]
fl_a4=amostra_paci4[['fl_severidade']]

corelacao=amostra_paci.corr(method ='spearman')
from sklearn.metrics import roc_auc_score
corelacao.to_csv('correlation_dados.csv')
x_train=x_train.fillna(0) 
x_test=x_test.fillna(0)
x_train.drop(['fl_severidade'],axis=1,inplace=True)
x_test.drop(['fl_severidade'],axis=1,inplace=True)
fl=amostra_paci['fl_severidade']
amostra_paci.drop(['fl_severidade'],axis=1,inplace=True) 
amostra_paci2.drop(['fl_severidade'],axis=1,inplace=True) 
amostra_paci3.drop(['fl_severidade'],axis=1,inplace=True) 
amostra_paci4.drop(['fl_severidade'],axis=1,inplace=True) 
y_train=y_train.fillna(0) 
y_test=y_test.fillna(0) 

print('Modelo Logistica')
#logistic=LogisticRegression(random_state=44)
#logistic.fit(x_train,y_train)
print(logistic(x_train,x_test,y_train,y_test,amostra_paci,fl,amostra_paci2,fl_a2,nome='log'))
print('')
print('')
print('Modelo Bayes Complement')
print(complement_bayes(x_train,x_test,y_train,y_test,amostra_paci,fl,amostra_paci3,fl_a3,nome='bayes'))
print('')
print('')
print('Modelo arvore')
print(arvore_dec(x_train,x_test,y_train,y_test,amostra_paci,fl,amostra_paci4,fl_a4,nome='arvor'))
print('')
