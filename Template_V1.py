# -*- coding: utf-8 -*-
#"""
#Created on Tue Mar 10 21:42:56 2020
#caminho -- C:\Users\fabio343\.spyder-py3
#@author: fabio343
#"""

from __future__ import division
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


### Definição de paramentros iniciais para o tratamento de dados
### Selecionando as informações presentes em arquivos .csv
### De modo a cruzar com outros arquivos para unificar,
### Em um unico dataframe os dados facilitando na manipulação.

dados_analises=pd.read_csv('teste_leitura_11.csv',sep=';',error_bad_lines=False)
dados_analises2=pd.read_csv('teste_leitura_12.csv',sep=';',error_bad_lines=False)
dados_analises3=pd.read_csv('teste_leitura_13.csv',sep=';',error_bad_lines=False)
dados_analises4=pd.read_csv('teste_leitura_14.csv',sep=';',error_bad_lines=False)
#dados_analises3=pd.read_csv('teste_leitura_11.csv',sep=';',error_bad_lines=False)
#dados_analises4=pd.read_csv('teste_leitura_11.csv',sep=';',error_bad_lines=False)
#dados_analises5=pd.read_csv('teste_leitura_11.csv',sep=';',error_bad_lines=False)
#dados_analises6=pd.read_csv('teste_leitura_11.csv',sep=';',error_bad_lines=False)
#dados_analises7=pd.read_csv('teste_leitura_11.csv',sep=';',error_bad_lines=False)
#dados_analises8=pd.read_csv('teste_leitura_11.csv',sep=';',error_bad_lines=False)

base_unificada= dados_analises.merge(dados_analises2, how='left', on=['Subject_ID'])
base_unificada1= base_unificada.merge(dados_analises3, how='left', on=['Subject_ID'])
base_unificada2= base_unificada1.merge(dados_analises4, how='left', on=['Subject_ID'])

#variaveis=base_unificada2.columns
#vetor_str=[]

#Verificando o tipo de dado presente e quais seriam as colunas.
A=base_unificada2.select_dtypes(include=['int64'])
B=base_unificada2.select_dtypes(include=['float64'])
C=base_unificada2.select_dtypes(include=['object'])

lista_trop=[]
#Removendo colunas que não trazem importancia para o modelo 
#Como indetificador de paciente,endereço, datas sem considerar data de nascimento.
for L in lista_trop: 
  base_unificada2.drop(L, axis=1, inplace=True)

#Removendo columnas com valores contantes

#Relizando a conversão de valores string para numericos
#Com o objetivo de simplificar a aplicação nas tecnicas de modelagem
#A alteração é dada com base na frequencia em relação com a resposta inicial desejada
#Ou seja inicialmente definimos pacientes com casos graves.
#Apos marcação realizamos um tipo de replace no qual 
#alteramos todos os campos com a nova marcação e salvamos os arquivos em formato pickle
for K in C.columns:
    tot_variavel=base_unificada2[[K]].count()
    marcacoes=base_unificada2[['targ',K]].query('targ=1').groupby(K).groupby('targ').count()/tot_variavel
    df1= base_unificada2.merge(marcacoes, how='left', on=[K])
    marcacoes.to_pickle(K+'.pickle')



#################################################################
#LIMPEZA DE VARIAVEIS CONSTANTES
##################################################################
vl_limpa_const=0.05 # variaveis com 95% dos campos repetidos
limpa_const = VarianceThreshold(threshold= vl_limpa_const)
limpa_const.fit(df1)
vars_const = [v for v in df1.columns if v not in  df1.columns[limpa_const.get_support()]]
qt_var = len([v for v in df1.columns if v not in  df1.columns[limpa_const.get_support()]])
 
print('Existem {} variaveis constantes com limite de {}'.format(qt_var,vl_limpa_const))
print('Variaveis constantes com limite de {}'.format(vl_limpa_const))
print(vars_const)
 
 
df1 = df1.drop(vars_const, axis = 1)
vars_const.to_pickle('vars_const.pickle')
base_unificada3=df1

#SALVANDO A BASE JÁ COM A LIMPEZA DE VARIAVEIS CONSTANTES PARA AGILIZAR O PROCESSO 
#DE CLASSIFICAÇÃO FINAL DO MODELO (DECIS)
base_unificada4=base_unificada3.copy()
#base_unificada3 = base_unificada2.loc[:,base_unificada2.apply(pd.Series.nunique) < 0.8]

#print('Covariancia')
#print(base_unificada3.cov())
#print()
#print(correlacao=base_unificada3.corr())
#print('correlacao')
#print()

#names =df.columns
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(correlacao, vmin=-1, vmax=1)
#fig.colorbar(cax)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
#plt.grid()
#plt.show()

#df2 = pd.read_pickle('my_df.pickle')
#tratando dados vazios adicionando um valor que não atribua peso no modelo
base_unificada3=base_unificada3.fillna(-9999999)     

#Selecionando amostra para modelagem após os tratamentos 
#Realizados em todos os dados
# E separando em treino e teste

amostra_paci =base_unificada3.sample(False,0.05,4321).write.csv('{}'.format('amostra_project'),header=True)
X_train, X_test = train_test_split(amostra_paci, test_size=0.33, random_state=42)

#Parametros para aplicação da modelagem (random forest)
#numero de amostras para uso no cross_validation da seleção de variaveis
qt_cv = 3
#numero de amostras para uso no cross_validation do desenvolvimento do modelo
qt_cv_2 = 4

#parametros para random forest sendo eles
#quantidade de arvores, profundidade das raizes e jobs
tuned_parameters = [{'n_estimators': [300,325,350,375,400,450,475,500,525,550,575,600],'max_depth': [3,4, 5, 6, 7,8,9],'n_jobs':[2,3,4,5,6]}]

#separando variavel targuet do dataframe para realizar a modelagem
#tanto para teste quanto para treinamento
y_train=df2['targuet'].copy()
del X_train['targuet']
X_train=X_train

y_test=X_test['targuet'].copy()
del X_test['targuet']
X_test=X_test


##################################################################
#LIMPEZA DE VARIAVEIS CORRELACIONADAS
##################################################################
vl_limpa_correl=0.8

corrmat = X_train.corr()
corrmat = corrmat.abs().unstack()
corrmat = corrmat.sort_values(ascending = False)
corrmat = corrmat[corrmat >= vl_limpa_correl]
corrmat = corrmat[corrmat < 1]
corrmat = pd.DataFrame(corrmat).reset_index()
corrmat.columns = ['var1','var2','corr']
corrmat.head()
 
 
grupo_vars = []
correlacao_grupo = []
 
for variavel in corrmat.var1.unique():
  if variavel not in grupo_vars:
    bloco_correl = corrmat[corrmat.var1 == variavel]
    grupo_vars = grupo_vars + list(bloco_correl.var2.unique()) + [variavel]
    
    correlacao_grupo.append(bloco_correl)
    
num_grupos = len(correlacao_grupo)    
print('temos {} grupos de correlacao'.format(num_grupos)) 
 
    
for grupo in correlacao_grupo:
  print(grupo)
  print()
  
vars_correl_drop = [] 
grupo = correlacao_grupo[3]
for grupo in correlacao_grupo:
  variaveis = list(grupo.var2.unique())+list(grupo.var1.unique())
  rf = RandomForestClassifier(n_estimators = 400,max_depth = 6, random_state = 0)
  rf.fit(X_train[variaveis],y_train)
  
  importancia_vars = pd.DataFrame(rf.feature_importances_,
                                  index = X_train[variaveis].columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
 
  
  vars_drop_grupo = importancia_vars.sort_values(ascending=False, by='importance')[1:]
  vars_drop_grupo = vars_drop_grupo.index.tolist()
  for var in vars_drop_grupo:
    vars_correl_drop.append(var)
  
list_drop_dup = [v for v in X_train.columns if v in  vars_correl_drop]
  

list_drop_dup.to_pickle('correlation.pickle') 
#
X_train = X_train.drop(list_drop_dup, axis = 1)
X_test = X_test.drop(list_drop_dup, axis = 1)
 
 
len(list(X_train.columns))
#Aplicação das tecnicas de modelagem apos o tratamento dos dados
modelo2= GridSearchCV(RandomForestClassifier(), tuned_parameters, cv= qt_cv, scoring='roc_auc')
modelo2.fit(X_train, y_train)

print('Melhores parametros')
print(modelo2.best_params_)
 
#modelo2 = RandomForestClassifier(n_estimators = n_estimators,max_depth = max_depth,n_jobs= num_jobs,random_state=0)
#modelo2.fit(X_train, y_train)

 
#Apos realizar a modelagem vemos a metrica como KS (
#um teste não paramétrico sobre a igualdade de distribuições de probabilidade 
#contínuas e unidimensionais que pode ser usado para comparar uma amostra com uma
#distribuição de probabilidade de referência (teste K–S uniamostral) ou duas 
#amostras uma com a outra (teste K–S biamostral)) e
#AUC (
#O AUC representa o grau ou medida de separabilidade. Quanto maior o AUC, 
#melhor o modelo está em prever 0s como 0s e 1s como 1s.
#Por exemplo, quanto maior a AUC, melhor o modelo está em distinguir
#entre pacientes com doença e pacientes sem doença)
#para verificar como
#Estar ocorrendo da descrição dos dados.

probs = np.round(modelo2.predict_proba(X_test.values)[:,1], 7)
fpr, tpr, thresholds = roc_curve(y_test, probs)
print('KS de Desenvolvimento: {0:.2f}% e AUC: {1:.2f}%'.format(stats.ks_2samp(fpr, tpr)[0] * 100.0, auc(fpr, tpr) * 100))
print((probs.max(),probs.min()))

 

probs = np.round(modelo2.predict_proba(X_train.values)[:,1], 7)
fpr, tpr, thresholds = roc_curve(y_train, probs)
print('KS de Validacao: {0:.2f}% e AUC: {1:.2f}%'.format(stats.ks_2samp(fpr, tpr)[0] * 100.0, auc(fpr, tpr) * 100))
print((probs.max(),probs.min()))

feature_importances = pd.DataFrame(modelo2.feature_importances_, index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances.count())

#Salvando modelo 
pickle.dump(modelo2,'modelo_random.pickle')

#---------------------
#Reprocessamento base inteira e aplico predict_proba na base inteira
#Determino os 4 grupos mais propensos e verifico se quem está lá condiz com 
#a caracteristica do tipo (severidade) da doença que esperamos

df_correlacionadas = pd.read_pickle('correlation.pickle')
modelo=pickle.load('modelo_random.pickle')

#Remove as mesmas variaveis que no processo de modelagem inicial
for k in df_correlacionadas:
    base_unificada4.drop(k)
    
#Remove as variaveis que sabemos não ser uteis para o modelo, mas dessa vez
#iremos manter o id do panciente para realizar as quebras dos 4 grupos
for L in lista_trop: 
  base_unificada4.drop(L, axis=1, inplace=True)
    
  
vetor=[]
for k in range(len(df_t1)):
  vetor_probabilidades.append(int(round(500+(log(1/(1-np.round(modelo.predict_proba(base_unificada4[1:].values)[:,1], 7)[k]))*20/log(2)),0)))

print(vetor_probabilidades)
df_vetor={'score':vetor_probabilidades}
df_vetor_final=pd.DataFrame(df_vetor,columns=['score'])
print(df_vetor_final)
print()
print(base_unificada4)
print()
base_unificada4['score']=df_vetor_final
base_unificada4=base_unificada4.sort_values('score',ascending=False)
quantil=pd.qcut(base_unificada4['score'],4)
base_score=base_unificada4.assign(quantil=quantil.values).reset_index()
print(base_score)

percentil=[]
numero=1

#Marcando os quartis ou seja a divisão nos 4 grupos de severidade
#considerando os valores de score e dessa forma teriamos:
#1 pacientes com caracteristicas de dengue severa
#de 2 em diante seria a redução do grau de severidade
#com base nisso verificar se o contexto biologico esta
#de fato fazendo sentindo
for k in range(len(base_score)-1):
    if base_score['quantil'][k+1]==base_score['quantil'][k]:
         percentil.append(numero)          
    elif numero==1:
        percentil.append(1)
        numero=numero+1
    elif numero==2:
        percentil.append(2)
        numero=numero+1
    elif numero==3:
        percentil.append(3)
        numero=numero+1
    else:
        percentil.append(4)
        
percentil.append(numero)

df_vetor_percentil=pd.DataFrame(percentil,columns=['quartil'])
base_score['quartil']=df_vetor_percentil
print()
print(base_score)




