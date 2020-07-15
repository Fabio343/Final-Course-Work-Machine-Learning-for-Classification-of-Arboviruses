# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:51:19 2020

@author: fabio343
"""

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
import category_encoders as ce   # version 1.2.8
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB



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

def plot_roc_curve(y_true, y_score, figsize=(10,6)):
    fpr, tpr,_ = roc_curve(y_true, y_score)
    plt.figure(figsize=figsize)
    auc_value = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def var_drop(df,colum):
  colunas=[]
  for k in colum:
     if df[k].sum()<=60:
       colunas.append(k)
  return colunas  


def logistic(x_train,x_test,y_train,y_test,X,fl,amostra_paci2):
    logistic=LogisticRegression(random_state=44)
    logistic.fit(x_train,y_train)
    pred=logistic.predict_proba(x_train)
    amostra_=logistic.predict_proba(amostra_paci2)
    amostra_paci2['probls']=0
    amostra_paci2['probls']=amostra_
    amostra_paci2.to_csv('modelo_logistic.csv')
    print('Treinamento AUC-ROC:{}'.format(roc_auc_score(y_train,pred[:,1])))
    pred_2=logistic.predict_proba(x_test)
    print('Validacao AUC-ROC:{}'.format(roc_auc_score(y_test,pred_2[:,1])))
    print(logistic.coef_)
    print(logistic.predict_proba(X))
    yhat = logistic.predict_proba(X)
    yhat = yhat[:, 1] 
    print(pd.crosstab(fl, logistic.predict(X)))
    print(classification_report(fl, logistic.predict(X)))
    print('AUC: %0.2f' % roc_auc_score(fl,yhat))
    plot_roc_curve(fl,yhat)
#esteve_hospitalizado_quantos_dias_7D,esteve_hospitalizado_quantos_dias_14D,laboratorio_hemograma_hematocrito,laboratorio_hemograma_hemoglobina
#tot_sinais_de_alarme,fl_sexo,idade

def complement_bayes(x_train,x_test,y_train,y_test,X,fl,amostra_paci3):
    Complement=ComplementNB()
    Complement.fit(x_train,y_train)
    pred=Complement.predict_proba(x_train)
    amostra_=Complement.predict_proba(amostra_paci3)
    amostra_paci3['probls']=0
    amostra_paci3['probls']=amostra_
    amostra_paci3.to_csv('modelo_complement_bayes.csv')
    print('Treinamento AUC-ROC:{}'.format(roc_auc_score(y_train,pred[:,1])))
    pred_2=Complement.predict_proba(x_test)
    print('Validacao AUC-ROC:{}'.format(roc_auc_score(y_test,pred_2[:,1])))
    print(Complement.predict_proba(X))
    yhat = Complement.predict_proba(X)
    yhat = yhat[:, 1] 
    print(pd.crosstab(fl, Complement.predict(X)))
    print(classification_report(fl, Complement.predict(X)))
    print('AUC: %0.2f' % roc_auc_score(fl,yhat))
    plot_roc_curve(fl,yhat)

### Definição de paramentros iniciais para o tratamento de dados
### Selecionando as informações presentes em arquivos .csv
### De modo a cruzar com outros arquivos para unificar,
### Em um unico dataframe os dados facilitando na manipulação.


dados_analises=pd.read_csv('pt_BR_0_arbobios2-20200309041123_v2.csv',sep=';',error_bad_lines=False)
dados_analises2=pd.read_csv('questionario_pt_BR_0_CRF_01 Critério de elegibilidade-20200309061346_v2.csv',sep=';',error_bad_lines=False)
dados_analises3=pd.read_csv('questionario_pt_BR_0_CRF_05 Dengue D7-20200309061441_v2.csv',sep=';',error_bad_lines=False)
dados_analises4=pd.read_csv('questionario_pt_BR_0_CRF_06 Dengue D14-20200309061459_v2.csv',sep=';',error_bad_lines=False)

#marcação das colunas para os pacientes de 7 e 14 dias 
for k in dados_analises3.columns:
    if k =='Subject_ID':
        dados_analises3=dados_analises3.rename(columns={k:k})
    elif k =='sexo':
        dados_analises3=dados_analises3.rename(columns={k:k})
    elif k=='esteve_hospitalizado_qual_hospital':
        dados_analises3=dados_analises3.rename(columns={k:k})
    else:     
         dados_analises3=dados_analises3.rename(columns={k:k+"_7D"})
    

for k in dados_analises4.columns:
    if k =='Subject_ID':
        dados_analises4=dados_analises4.rename(columns={k:k})
    elif k =='sexo':
        dados_analises4=dados_analises4.rename(columns={k:k})
    elif k=='esteve_hospitalizado_qual_hospital':
        dados_analises4=dados_analises4.rename(columns={k:k})
    else:     
         dados_analises4=dados_analises4.rename(columns={k:k+"_14D"})
 
#união de todas as bases via merge para a base unificada
# considerando como chave o id o paciente         
colunas1= dados_analises2.columns.difference(dados_analises.columns)
col=[]
col.append('Subject_ID')
for k in colunas1:
  col.append(k)
base_unificada=dados_analises.merge(dados_analises2[col], how='left',on=['Subject_ID'])


colunas2= dados_analises3.columns.difference(base_unificada.columns)
col=[]
col.append('Subject_ID')
for k in colunas2:
  col.append(k)
base_unificada2=base_unificada.merge(dados_analises3[col], how='left',on=['Subject_ID'])

colunas3= dados_analises4.columns.difference(base_unificada2.columns)
col=[]
col.append('Subject_ID')
for k in colunas3:
  col.append(k)
base_unificada3=base_unificada2.merge(dados_analises4[col], how='left',on=['Subject_ID'])

###############################################################################
dados_analises5=pd.read_csv('questionario_pt_BR_0_CRF_03 Inclusão não gestante-20200309061411_v2.csv',sep=';',error_bad_lines=False)
dados_analises6=pd.read_csv('arbobios_erika_dez2019.csv',sep=';',error_bad_lines=False)

dados_analises6=dados_analises6[['Subject_ID','CLASSIFICACAO FINAL']]

ids=dados_analises6[['Subject_ID']]

colunas4= dados_analises5.columns.difference(base_unificada3.columns)
col=[]
col.append('Subject_ID')
for k in colunas4:
  col.append(k)
base_unificada4=base_unificada3.merge(dados_analises5[col], how='left',on=['Subject_ID']) 

colunas5= dados_analises6.columns.difference(base_unificada4.columns)
col=[]
col.append('Subject_ID')
for k in colunas5:
  col.append(k)
base_unificada5=base_unificada4.merge(dados_analises6[col],how='left',on=['Subject_ID'])

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


lista_trop=[
"Hospital",
"id_mvf",
"autoriza_contato_mensagem",
"exam1_sample_id",
"exam1_date_exam",
"exam2_sample_id",
"exam2_date_exam",
"exam3_sample_id",
"exam3_date_exam",
"exam4_sample_id",
"exam4_date_exam",
"exam5_sample_id",
"exam5_date_exam",
"exam6_sample_id",
"exam6_date_exam",
"exam7_sample_id",
"exam7_date_exam",
"_1_refused_Created_Date",
"_1_DENV_SA_Created_Date",
"_1_DENV_DAA_Created_Date",
"_1_CKIKV_Created_Date",
"_1_ZIKV_Created_Date",
"_1_TCLE_Created_Date",
"_1_Data_tcle",
"_1_qex_triagem_Created_Date",
"_1_data_qex_triagem",
"_1_Qex_Inclusao_Created_Date",
"_1_data_qex_inclusao",
"_1_Coleta_Sangue_Inscricao_Created_Date",
"_1_Coleta_Sangue_Inscricao.Collect_Date",
"_1_Coleta_Sangue_Inscricao.Collect_Hour",
"_1_Coleta_Sangue_Inscricao.Collect_Minute",
"_1_Coleta_Sangue_Inscricao.Label_set",
"_2_Coleta_Sangue_Inscricao_Created_Date",
"_2_Coleta_Sangue_Inscricao.Collect_Date",
"_2_Coleta_Sangue_Inscricao.Collect_Hour",
"_2_Coleta_Sangue_Inscricao.Collect_Minute",
"_2_Coleta_Sangue_Inscricao.Label_set",
"_3_Coleta_Sangue_Inscricao_Created_Date",
"_3_Coleta_Sangue_Inscricao.Collect_Date",
"_3_Coleta_Sangue_Inscricao.Collect_Hour",
"_3_Coleta_Sangue_Inscricao.Collect_Minute",
"_3_Coleta_Sangue_Inscricao.Label_set",
"_1_Teste_rapido_agns1_denv_Created_Date",
"_1_data-agns1",
"_1_teste_rapido_IgM_DENV_Created_Date",
"_1_data_teste_rapido_IgM_DENV",
"_1_teste_rapido_gravidez_Created_Date",
"_1_data_teste_rapido_gravidez",
"_1_TCLE_Biobanco_Created_Date",
"_1_data_TCLE_Biobanco",
"_1_CHIKV_PCR _Request_Created_Date",
"_1_Date_CHIKV_PCR_Request",
"_1_ZIKV_PCR_Request_Created_Date",
"_1_Date_ZIKV_PCR_Request",
"_1_DENV_PCR _Request_Created_Date",
"_1_HCGQT_request_Created_Date",
"_1_TGO_request_Created_Date",
"_1_TGP_request_Created_Date",
"_1_no_collection_reason_Created_Date",
"_1_date_reason_no_collection",
"_1_contato_tel",
"_1_contato_tel_Created_Date",
"_1_data_cont_tel",
"_1_time_cont_tel",
"_1_res_cont_tel",
"_1_motivo_cont_tel_sem_sucesso",
"_1_outros_cont_tel_sem_sucesso",
"_1_obs_cont_tel_",
"_2_contato_tel",
"_2_contato_tel_Created_Date",
"_2_data_cont_tel",
"_2_time_cont_tel",
"_2_res_cont_tel",
"_2_motivo_cont_tel_sem_sucesso",
"_2_outros_cont_tel_sem_sucesso",
"_2_obs_cont_tel_",
"_3_contato_tel",
"_3_contato_tel_Created_Date",
"_3_data_cont_tel",
"_3_time_cont_tel",
"_3_res_cont_tel",
"_3_motivo_cont_tel_sem_sucesso",
"_3_outros_cont_tel_sem_sucesso",
"_3_obs_cont_tel_",
"_4_contato_tel",
"_4_contato_tel_Created_Date",
"_4_data_cont_tel",
"_4_time_cont_tel",
"_4_res_cont_tel",
"_4_motivo_cont_tel_sem_sucesso",
"_4_outros_cont_tel_sem_sucesso",
"_4_obs_cont_tel_",
"_5_contato_tel",
"_5_contato_tel_Created_Date",
"_5_data_cont_tel",
"_5_time_cont_tel",
"_5_res_cont_tel",
"_5_motivo_cont_tel_sem_sucesso",
"_5_outros_cont_tel_sem_sucesso",
"_5_obs_cont_tel_",
"_6_contato_tel",
"_6_contato_tel_Created_Date",
"_6_data_cont_tel",
"_6_time_cont_tel",
"_6_res_cont_tel",
"_6_motivo_cont_tel_sem_sucesso",
"_6_outros_cont_tel_sem_sucesso",
"_6_obs_cont_tel_",
"_7_contato_tel",
"_7_contato_tel_Created_Date",
"_7_data_cont_tel",
"_7_time_cont_tel",
"_7_res_cont_tel",
"_7_motivo_cont_tel_sem_sucesso",
"_7_outros_cont_tel_sem_sucesso",
"_7_obs_cont_tel_",
"_8_contato_tel",
"_8_contato_tel_Created_Date",
"_8_data_cont_tel",
"_8_time_cont_tel",
"_8_res_cont_tel",
"_8_motivo_cont_tel_sem_sucesso",
"_8_outros_cont_tel_sem_sucesso",
"_8_obs_cont_tel_",
"_9_contato_tel",
"_9_contato_tel_Created_Date",
"_9_data_cont_tel",
"_9_time_cont_tel",
"_9_res_cont_tel",
"_9_motivo_cont_tel_sem_sucesso",
"_9_outros_cont_tel_sem_sucesso",
"_9_obs_cont_tel_",
"_10_contato_tel",
"_10_contato_tel_Created_Date",
"_10_data_cont_tel",
"_10_time_cont_tel",
"_10_res_cont_tel",
"_10_motivo_cont_tel_sem_sucesso",
"_10_outros_cont_tel_sem_sucesso",
"_10_obs_cont_tel_",
"_11_contato_tel",
"_11_contato_tel_Created_Date",
"_11_data_cont_tel",
"_11_time_cont_tel",
"_11_res_cont_tel",
"_11_motivo_cont_tel_sem_sucesso",
"_11_outros_cont_tel_sem_sucesso",
"_11_obs_cont_tel_",
"_12_contato_tel",
"_12_contato_tel_Created_Date",
"_12_data_cont_tel",
"_12_time_cont_tel",
"_12_res_cont_tel",
"_12_motivo_cont_tel_sem_sucesso",
"_12_outros_cont_tel_sem_sucesso",
"_12_obs_cont_tel_",
"_13_contato_tel",
"_13_contato_tel_Created_Date",
"_13_data_cont_tel",
"_13_time_cont_tel",
"_13_res_cont_tel",
"_13_motivo_cont_tel_sem_sucesso",
"_13_outros_cont_tel_sem_sucesso",
"_13_obs_cont_tel_",
"_14_contato_tel",
"_14_contato_tel_Created_Date",
"_14_data_cont_tel",
"_14_time_cont_tel",
"_14_res_cont_tel",
"_14_motivo_cont_tel_sem_sucesso",
"_14_outros_cont_tel_sem_sucesso",
"_14_obs_cont_tel_",
"_15_contato_tel",
"_15_contato_tel_Created_Date",
"_15_data_cont_tel",
"_15_time_cont_tel",
"_15_res_cont_tel",
"_15_motivo_cont_tel_sem_sucesso",
"_15_outros_cont_tel_sem_sucesso",
"_15_obs_cont_tel_",
"_16_contato_tel",
"_16_contato_tel_Created_Date",
"_16_data_cont_tel",
"_16_time_cont_tel",
"_16_res_cont_tel",
"_16_motivo_cont_tel_sem_sucesso",
"_16_outros_cont_tel_sem_sucesso",
"_16_obs_cont_tel_",
"_17_contato_tel",
"_17_contato_tel_Created_Date",
"_17_data_cont_tel",
"_17_time_cont_tel",
"_17_res_cont_tel",
"_17_motivo_cont_tel_sem_sucesso",
"_17_outros_cont_tel_sem_sucesso",
"_17_obs_cont_tel_",
"_18_contato_tel",
"_18_contato_tel_Created_Date",
"_18_data_cont_tel",
"_18_time_cont_tel",
"_18_res_cont_tel",
"_18_motivo_cont_tel_sem_sucesso",
"_18_outros_cont_tel_sem_sucesso",
"_18_obs_cont_tel_",
"_19_contato_tel",
"_19_contato_tel_Created_Date",
"_19_data_cont_tel",
"_19_time_cont_tel",
"_19_res_cont_tel",
"_19_motivo_cont_tel_sem_sucesso",
"_19_outros_cont_tel_sem_sucesso",
"_19_obs_cont_tel_",
"_20_contato_tel",
"_20_contato_tel_Created_Date",
"_20_data_cont_tel",
"_20_time_cont_tel",
"_20_res_cont_tel",
"_20_motivo_cont_tel_sem_sucesso",
"_20_outros_cont_tel_sem_sucesso",
"_20_obs_cont_tel_",
"_21_contato_tel",
"_21_contato_tel_Created_Date",
"_21_data_cont_tel",
"_21_time_cont_tel",
"_21_res_cont_tel",
"_21_motivo_cont_tel_sem_sucesso",
"_21_outros_cont_tel_sem_sucesso",
"_21_obs_cont_tel_",
"_22_contato_tel",
"_22_contato_tel_Created_Date",
"_22_data_cont_tel",
"_22_time_cont_tel",
"_22_res_cont_tel",
"_22_motivo_cont_tel_sem_sucesso",
"_22_outros_cont_tel_sem_sucesso",
"_22_obs_cont_tel_",
"_23_contato_tel",
"_23_contato_tel_Created_Date",
"_23_data_cont_tel",
"_23_time_cont_tel",
"_23_res_cont_tel",
"_23_motivo_cont_tel_sem_sucesso",
"_23_outros_cont_tel_sem_sucesso",
"_23_obs_cont_tel_",
"_24_contato_tel",
"_24_contato_tel_Created_Date",
"_24_data_cont_tel",
"_24_time_cont_tel",
"_24_res_cont_tel",
"_24_motivo_cont_tel_sem_sucesso",
"_24_outros_cont_tel_sem_sucesso",
"_24_obs_cont_tel_",
"_25_contato_tel",
"_25_contato_tel_Created_Date",
"_25_data_cont_tel",
"_25_time_cont_tel",
"_25_res_cont_tel",
"_25_motivo_cont_tel_sem_sucesso",
"_25_outros_cont_tel_sem_sucesso",
"_25_obs_cont_tel_",
"_26_contato_tel",
"_26_contato_tel_Created_Date",
"_26_data_cont_tel",
"_26_time_cont_tel",
"_26_res_cont_tel",
"_26_motivo_cont_tel_sem_sucesso",
"_26_outros_cont_tel_sem_sucesso",
"_26_obs_cont_tel_",
"_27_contato_tel",
"_27_contato_tel_Created_Date",
"_27_data_cont_tel",
"_27_time_cont_tel",
"_27_res_cont_tel",
"_27_motivo_cont_tel_sem_sucesso",
"_27_outros_cont_tel_sem_sucesso",
"_27_obs_cont_tel_",
"_28_contato_tel",
"_28_contato_tel_Created_Date",
"_28_data_cont_tel",
"_28_time_cont_tel",
"_28_res_cont_tel",
"_28_motivo_cont_tel_sem_sucesso",
"_28_outros_cont_tel_sem_sucesso",
"_28_obs_cont_tel_",
"_29_contato_tel",
"_29_contato_tel_Created_Date",
"_29_data_cont_tel",
"_29_time_cont_tel",
"_29_res_cont_tel",
"_29_motivo_cont_tel_sem_sucesso",
"_29_outros_cont_tel_sem_sucesso",
"_29_obs_cont_tel_",
"_30_contato_tel",
"_30_contato_tel_Created_Date",
"_30_data_cont_tel",
"_30_time_cont_tel",
"_30_res_cont_tel",
"_30_motivo_cont_tel_sem_sucesso",
"_30_outros_cont_tel_sem_sucesso",
"_30_obs_cont_tel_",
"_31_contato_tel",
"_31_contato_tel_Created_Date",
"_31_data_cont_tel",
"_31_time_cont_tel",
"_31_res_cont_tel",
"_31_motivo_cont_tel_sem_sucesso",
"_31_outros_cont_tel_sem_sucesso",
"_31_obs_cont_tel_",
"_32_contato_tel",
"_32_contato_tel_Created_Date",
"_32_data_cont_tel",
"_32_time_cont_tel",
"_32_res_cont_tel",
"_32_motivo_cont_tel_sem_sucesso",
"_32_outros_cont_tel_sem_sucesso",
"_32_obs_cont_tel_",
"_33_contato_tel",
"_33_contato_tel_Created_Date",
"_33_data_cont_tel",
"_33_time_cont_tel",
"_33_res_cont_tel",
"_33_motivo_cont_tel_sem_sucesso",
"_33_outros_cont_tel_sem_sucesso",
"_33_obs_cont_tel_",
"_34_contato_tel",
"_34_contato_tel_Created_Date",
"_34_data_cont_tel",
"_34_time_cont_tel",
"_34_res_cont_tel",
"_34_motivo_cont_tel_sem_sucesso",
"_34_outros_cont_tel_sem_sucesso",
"_34_obs_cont_tel_",
"_35_contato_tel",
"_35_contato_tel_Created_Date",
"_35_data_cont_tel",
"_35_time_cont_tel",
"_35_res_cont_tel",
"_35_motivo_cont_tel_sem_sucesso",
"_35_outros_cont_tel_sem_sucesso",
"_35_obs_cont_tel_",
"_36_contato_tel",
"_36_contato_tel_Created_Date",
"_36_data_cont_tel",
"_36_time_cont_tel",
"_36_res_cont_tel",
"_36_motivo_cont_tel_sem_sucesso",
"_36_outros_cont_tel_sem_sucesso",
"_36_obs_cont_tel_",
"_37_contato_tel",
"_37_contato_tel_Created_Date",
"_37_data_cont_tel",
"_37_time_cont_tel",
"_37_res_cont_tel",
"_37_motivo_cont_tel_sem_sucesso",
"_37_outros_cont_tel_sem_sucesso",
"_37_obs_cont_tel_",
"_38_contato_tel",
"_38_contato_tel_Created_Date",
"_38_data_cont_tel",
"_38_time_cont_tel",
"_38_res_cont_tel",
"_38_motivo_cont_tel_sem_sucesso",
"_38_outros_cont_tel_sem_sucesso",
"_38_obs_cont_tel_",
"_39_contato_tel",
"_39_contato_tel_Created_Date",
"_39_data_cont_tel",
"_39_time_cont_tel",
"_39_res_cont_tel",
"_39_motivo_cont_tel_sem_sucesso",
"_39_outros_cont_tel_sem_sucesso",
"_39_obs_cont_tel_",
"_1_visita_agendada_Created_Date",
"_1_data_visita_agendada",
"_1_hora_visita_agendada",
"_1_minutos_visita_agendada",
"_2_visita_agendada_Created_Date",
"_2_data_visita_agendada",
"_2_hora_visita_agendada",
"_2_minutos_visita_agendada",
"_3_visita_agendada_Created_Date",
"_3_data_visita_agendada",
"_3_hora_visita_agendada",
"_3_minutos_visita_agendada",
"_4_visita_agendada_Created_Date",
"_4_data_visita_agendada",
"_4_hora_visita_agendada",
"_4_minutos_visita_agendada",
"_1_visita_Created_Date",
"_1_data_visita",
"_2_visita_Created_Date",
"_2_data_visita",
"_3_visita_Created_Date",
"_3_data_visita",
"_4_visita_Created_Date",
"_4_data_visita",
"_1_coleta_sangue_retorno_DENV_Created_Date",
"_1_coleta_sangue_retorno_DENV.Collect_Date",
"_1_coleta_sangue_retorno_DENV.Collect_Hour",
"_1_coleta_sangue_retorno_DENV.Collect_Minute",
"_1_coleta_sangue_retorno_DENV.Label_set",
"_2_coleta_sangue_retorno_DENV_Created_Date",
"_2_coleta_sangue_retorno_DENV.Collect_Date",
"_2_coleta_sangue_retorno_DENV.Collect_Hour",
"_2_coleta_sangue_retorno_DENV.Collect_Minute",
"_2_coleta_sangue_retorno_DENV.Label_set",
"_1_coleta_sangue_retorno_CHIKV_Created_Date",
"_1_coleta_sangue_retorno_CHIKV.Collect_Date",
"_1_coleta_sangue_retorno_CHIKV.Collect_Hour",
"_1_coleta_sangue_retorno_CHIKV.Collect_Minute",
"_1_coleta_sangue_retorno_CHIKV.Label_set",
"_2_coleta_sangue_retorno_CHIKV_Created_Date",
"_2_coleta_sangue_retorno_CHIKV.Collect_Date",
"_2_coleta_sangue_retorno_CHIKV.Collect_Hour",
"_2_coleta_sangue_retorno_CHIKV.Collect_Minute",
"_2_coleta_sangue_retorno_CHIKV.Label_set",
"_3_coleta_sangue_retorno_CHIKV_Created_Date",
"_3_coleta_sangue_retorno_CHIKV.Collect_Date",
"_3_coleta_sangue_retorno_CHIKV.Collect_Hour",
"_3_coleta_sangue_retorno_CHIKV.Collect_Minute",
"_3_coleta_sangue_retorno_CHIKV.Label_set",
"_1_coleta _seg_gest_15d_Created_Date",
"_1_coleta _seg_gest_15d.Collect_Date",
"_1_coleta _seg_gest_15d.Collect_Hour",
"_1_coleta _seg_gest_15d.Collect_Minute",
"_1_coleta _seg_gest_15d.Label_set",
"_1_coleta_sangue_retorno_gestantes_Created_Date",
"_1_coleta_sangue_retorno_gestantes.Collect_Date",
"_1_coleta_sangue_retorno_gestantes.Collect_Hour",
"_1_coleta_sangue_retorno_gestantes.Collect_Minute",
"_1_coleta_sangue_retorno_gestantes.Label_set",
"_1_qex_denv_seg_7_Created_Date",
"_1_data_qex_seg_denv_7",
"_1_qex_seg_denv_14_Created_Date",
"_1_data_qex_seg_denv_14",
"_1_qex_seg_chikv_21_Created_Date",
"_1_data_qex_seg_chikv_14",
"_1_qex_seg_chikv_90_Created_Date",
"_1_data_qex_seg_chikv_90",
"_1_qex_seg_chikv_90_faltosos_Created_Date",
"_1_data_qex_seg_chikv_90_faltosos",
"_1_qex_seguimento_gestante_Created_Date",
"_1_data_qex_seg_gestante",
"_1_encaminhamento_ultrassom_Created_Date",
"_1_Data_encaminhamento_ultrasson",
"_1_qex_ultrassom_chikv_Created_Date",
"_1_data_qex_ultrassom",
"_2_qex_ultrassom_chikv_Created_Date",
"_2_data_qex_ultrassom",
"_1_ultrassom_imagem_chikv_Created_Date",
"_1_data_ultrassom_imagem",
"_1_qex_ultrasson_gestantes_Created_Date",
"_1_data_qex_ultrassom_gestantes",
"_1_ultrassom_imagem_gestantes_Created_Date",
"_1_qex_nasc_gestantes_Created_Date",
"_1_data_qex_nasc_gestantes",
"_1_seguimento_incompleto_Created_Date",
"_1_data_seguimento_incompleto",
"_1_dados_obito_Created_Date",
"_1_data_obito",
"_1_qex_perda_seg_zikv_Created_Date",
"_1_data_perda acomp_zikv",
"_1_inelegivel_Created_Date",
"_1_Elegivel_Created_Date",
"_1_Gestante_Created_Date",
"_1_agend_visitas_Created_Date",
"_1_Result_POS_denv_Created_Date",
"_1_Result_POS_chikv_Created_Date",
"_1_Result_POS_zikv_Created_Date",
"_1_Result_NEG_Created_Date",
"_1_PCR_CHIKV_Positive_Result_Created_Date",
"_1_date_PCR_CHIKV_Result",
"_1_PCR_CHIKV_Negative_Result_Created_Date",
"_1_PCR_ZIKV_Positive_Result_Created_Date",
"_1_date_PCR_ZIKV_Result",
"_1_PCR_ZIKV_Negative_Result_Created_Date",
"_1_PCR_DENV_Positive_Result_Created_Date",
"_1_date_PCR_DENV_Result",
"_1_PCR_DENV_Negative_Result_Created_Date",
"_1_falta_visita_agendada_Created_Date",
"_1_date_missed_appointment",
"_1_qex_triagem_finalizado_Created_Date",
"_1_prazo_ultrapassado_denv7_Created_Date",
"_1_qex_inclusao-finalizado_Created_Date",
"_1_prazo_ultrapassado_denv14_Created_Date",
"_1_prazo_ultrapassado_chikv21_Created_Date",
"_1_prazo_ultrapassado_chikv90_Created_Date",
"_1_prazo_ultrapassado_gestante_Created_Date",
"_1_qex_denv_d7_finalizado_Created_Date",
"_1_qex_denv_d14_finalizado_Created_Date",
"_1_qex_chikv_d21_finalizado_Created_Date",
"_1_qex_chikv_d90_finalizado_Created_Date",
"_1_qex_seg_gest_finalizado_Created_Date",
"_1_qex_nasc_gest_finalizado_Created_Date",
"_1_denv_retornos_completos_Created_Date",
"_1_chikv_retornos_completos_Created_Date",
"_1_ZIKV_Retornos_completos_Created_Date",
"_1_DENV_SA_Retornos_Completos_Created_Date",
"_1_due_scheduled_period_Created_Date",
"_1_TGO_Result_Created_Date",
"_1_TGP_Result_Created_Date",
"_1_HCGQT_Result_Created_Date",
"_1_date_refusal",
"esteve_hospitalizado_qual_hospital",
"laboratorio_tgo",
"laboratorio_tgp",
"laboratorio_hemograma_plaquetas"]

#"laboratorio_tgo",
#"laboratorio_tgp",
#"laboratorio_hemograma_plaquetas",
#"laboratorio_hemograma_hemoglobina",
#"laboratorio_hemograma_hematocrito"

for L in lista_trop: 
  base_unificada5.drop(L, axis=1, inplace=True)
base_unificada5.to_csv('base1.csv')

# processo que usamos para separar variaveis com mais de 1 informação
# dentro de uma mesma celula da base 
# com essa separação podemos criar flags para cada sintoma
# deixando mais claro o impacto de cada sintoma no 
# processo final
 
base_unificada5=base_unificada5.assign(sinais_alarme_qual=base_unificada5.sinais_alarme_qual.str.split('-')).explode('sinais_alarme_qual').reset_index(drop=True).drop_duplicates('Subject_ID')
base_unificada5['sinais_alarme_qual']=base_unificada5['sinais_alarme_qual'].str.strip()
base_dummy1= pd.get_dummies(base_unificada5['sinais_alarme_qual'], prefix='SAQ')
base_unificada5=base_unificada5.merge(base_dummy1,left_index=True,right_index=True)
             
base_unificada5=base_unificada5.assign(sinais_de_alarme_7D=base_unificada5.sinais_de_alarme_7D.str.split('-')).explode('sinais_de_alarme_7D').reset_index(drop=True).drop_duplicates('Subject_ID')
base_unificada5['sinais_de_alarme_7D']=base_unificada5['sinais_de_alarme_7D'].str.strip()
base_dummy2= pd.get_dummies(base_unificada5['sinais_de_alarme_7D'],prefix='SA7D')
base_unificada5=base_unificada5.merge(base_dummy2,left_index=True,right_index=True)

base_unificada5=base_unificada5.assign(sinais_de_alarme_14D=base_unificada5.sinais_de_alarme_14D.str.split('-')).explode('sinais_de_alarme_14D').reset_index(drop=True).drop_duplicates('Subject_ID')
base_unificada5['sinais_de_alarme_14D']=base_unificada5['sinais_de_alarme_14D'].str.strip()
base_dummy3= pd.get_dummies(base_unificada5['sinais_de_alarme_14D'],prefix='SA14D')
base_unificada5=base_unificada5.merge(base_dummy3,left_index=True,right_index=True)

base_unificada5=base_unificada5.assign(sintoma_febre_7dias=base_unificada5.sintoma_febre_7dias.str.split('-')).explode('sintoma_febre_7dias').reset_index(drop=True).drop_duplicates('Subject_ID')
base_unificada5['sintoma_febre_7dias']=base_unificada5['sintoma_febre_7dias'].str.strip()
base_dummy4= pd.get_dummies(base_unificada5['sintoma_febre_7dias'],prefix='SF7D')
base_unificada5=base_unificada5.merge(base_dummy4,left_index=True,right_index=True)

base_unificada5=base_unificada5.assign(outros_sinais_alarme_quais=base_unificada5.outros_sinais_alarme_quais.str.split('-')).explode('outros_sinais_alarme_quais').reset_index(drop=True).drop_duplicates('Subject_ID')
base_unificada5['outros_sinais_alarme_quais']=base_unificada5['outros_sinais_alarme_quais'].str.strip()
base_dummy5= pd.get_dummies(base_unificada5['outros_sinais_alarme_quais'],prefix='OSAQ')
base_unificada5=base_unificada5.merge(base_dummy5,left_index=True,right_index=True)

variaveis_dummy_2=[]
for k in ['dor_articulacao_lado', 'dor_articulacao_local',
       'dor_articulacao_qtde', 'dor_articulacao_rigidez', 'manchas_pele_cocam',
       'manchas_pele_cor','manchas_pele_local', 'sangramento',
       'sangramento_como','sintoma_febre_7dias_artrite',
       'sintoma_febre_7dias_conjuntivite', 'sintoma_febre_7dias_inchaco',
       'esteve_hospitalizado_7D', 'esteve_hospitalizado_agravamento_dengue_7D',
       'esteve_hospitalizado_motivo_7D',
       'esteve_hospitalizado_14D','esteve_hospitalizado_agravamento_dengue_14D',
       'esteve_hospitalizado_motivo_14D','edema', 'edema_onde',
       'exantema', 'exantema_onde', 'ja_teve_chikungunya', 'ja_teve_dengue',
       'ja_teve_zika','outros_sinais_alarme', 'raca',
       'sinais_artrite', 'sinais_artrite_onde',
       'sinais_artrite_onde_apendicular', 'tem_alguma_doenca_das_articulacoes']:
  base_dummy_novas= pd.get_dummies(base_unificada5[k],prefix=k)
  base_unificada5=base_unificada5.merge(base_dummy_novas,left_index=True,right_index=True)
 
  for i in base_dummy_novas.columns:
    variaveis_dummy_2.append(i)
    
base_unificada5.drop('sinais_alarme_qual', axis=1, inplace=True)
base_unificada5.drop('sinais_de_alarme_7D', axis=1, inplace=True)
base_unificada5.drop('sinais_de_alarme_14D', axis=1, inplace=True)
base_unificada5.drop('sintoma_febre_7dias', axis=1, inplace=True)
base_unificada5.drop('outros_sinais_alarme_quais', axis=1, inplace=True)

for L in ['dor_articulacao_lado', 'dor_articulacao_local',
       'dor_articulacao_qtde', 'dor_articulacao_rigidez', 'manchas_pele_cocam',
       'manchas_pele_cor','manchas_pele_local', 'sangramento',
       'sangramento_como','sintoma_febre_7dias_artrite',
       'sintoma_febre_7dias_conjuntivite', 'sintoma_febre_7dias_inchaco',
       'esteve_hospitalizado_7D', 'esteve_hospitalizado_agravamento_dengue_7D',
       'esteve_hospitalizado_motivo_7D',
       'esteve_hospitalizado_14D','esteve_hospitalizado_agravamento_dengue_14D',
       'esteve_hospitalizado_motivo_14D','edema', 'edema_onde',
       'exantema', 'exantema_onde', 'ja_teve_chikungunya', 'ja_teve_dengue',
       'ja_teve_zika','outros_sinais_alarme', 'raca',
       'sinais_artrite', 'sinais_artrite_onde',
       'sinais_artrite_onde_apendicular', 'tem_alguma_doenca_das_articulacoes']: 
  base_unificada5.drop(L, axis=1, inplace=True)

base_unificada5.reset_index()
base_unificada5.to_csv('base2.csv')

variaveis_dummy=[]
for k in base_dummy1.columns:
    variaveis_dummy.append(k)

for k in base_dummy2.columns:
    variaveis_dummy.append(k)

for k in base_dummy3.columns:
    variaveis_dummy.append(k)

for k in base_dummy4.columns:
    variaveis_dummy.append(k)

for k in base_dummy5.columns:
    variaveis_dummy.append(k)
    
 
base_unificada5['tot_sinais_de_alarme']=0
base_unificada5['tot_sinais_de_alarme']=base_unificada5[variaveis_dummy].sum(axis=1)    

#Marcando a flag para a modelagem considerando 1 todos os casos severos e 0
#casos não severos 

base_unificada5['fl_severidade'] = 0
base_unificada5['fl_severidade'][(base_unificada5['CLASSIFICACAO FINAL']==3.0)] = 1
base_unificada5['fl_sexo'] = 0
base_unificada5['fl_sexo'][(base_unificada5['sexo']=='feminino')] = 1

import datetime
base_unificada5['DOB']=pd.to_datetime(base_unificada5['DOB'],errors='coerce').replace(np.nan,datetime.datetime.now())
base_unificada5['febre_7dias_quando']=pd.to_datetime(base_unificada5['febre_7dias_quando'],errors='coerce').replace(np.nan,datetime.datetime.now())

base_unificada5=base_unificada5.reset_index()
base_unificada5['idade'] = 0
for L in range(len(base_unificada5['DOB'])):
     difff=base_unificada5['febre_7dias_quando'][L]-base_unificada5['DOB'][L]
     base_unificada5['idade'][L]=int((difff.days + difff.seconds/86400)/365.2425)
 
#Filtrando apenas os pacientes com dengue as demais
#aborviroses serão trabalhadas em outra etapa do processo
base_unificada5_filtrada=base_unificada5[base_unificada5['exam1_agent'] =='DENV'].reset_index()
base_unificada5_filtrada=base_unificada5[(base_unificada5['exam1_resultado_desc'] =='Reactive') | (base_unificada5['exam2_resultado_desc'] =='Reactive') | (base_unificada5['exam3_resultado_desc'] =='Reactive') | (base_unificada5['exam4_resultado_desc'] =='Reactive') | (base_unificada5['exam5_resultado_desc'] =='Reactive') | (base_unificada5['exam6_resultado_desc'] =='Reactive')].reset_index()

#Relizando a conversão de valores string para numericos
#Com o objetivo de simplificar a aplicação nas tecnicas de modelagem
#A alteração é dada com base na frequencia em relação com a resposta inicial desejada
#Ou seja inicialmente definimos pacientes com casos graves.
#Apos marcação realizamos um tipo de replace no qual 
#alteramos todos os campos com a nova marcação e salvamos os arquivos em formato pickle

base_unificada5_filtrada.drop('sexo', axis=1, inplace=True)
base_unificada5_filtrada.drop('DOB', axis=1, inplace=True)
base_unificada5_filtrada.drop('febre_7dias_quando', axis=1, inplace=True)
base_unificada5_filtrada.drop('index', axis=1, inplace=True)
base_unificada5_filtrada.drop('CLASSIFICACAO FINAL', axis=1, inplace=True)
C=base_unificada5_filtrada.select_dtypes(include=['object'])

base_unificada5_filtrada.to_csv('base3.csv')

lista_drop_2=[
"gestante",
"exam1_agent",
"exam1_techinic",
"exam1_resultado_desc",
"exam1_sorotipo_desc",
"exam2_agent",
"exam2_techinic",
"exam2_resultado_desc",
"exam2_sorotipo_desc",
"exam3_agent",
"exam3_techinic",
"exam3_resultado_desc",
"exam3_sorotipo_desc",
"exam4_agent",
"exam4_techinic",
"exam4_resultado_desc",
"exam4_sorotipo_desc",
"exam5_agent",
"exam5_techinic",
"exam5_resultado_desc",
"exam5_sorotipo_desc",
"exam6_agent",
"exam6_techinic",
"exam6_resultado_desc",
"exam6_sorotipo_desc",
"exam7_agent",
"exam7_techinic",
"exam7_resultado_desc",
"exam7_sorotipo_desc",
"_1_refused",
"_1_refusal_reason",
"_1_outro_recusa",
"_1_DENV_SA",
"_1_DENV_DAA",
"_1_CKIKV",
"_1_ZIKV",
"_1_TCLE",
"_1_assentment",
"_1_qex_triagem",
"_1_Qex_Inclusao",
"_1_Coleta_Sangue_Inscricao",
"_2_Coleta_Sangue_Inscricao",
"_3_Coleta_Sangue_Inscricao",
"_1_Teste_rapido_agns1_denv",
"_1_resultado_agns1",
"_1_teste_rapido_IgM_DENV",
"_1_resultado_teste_rapido_IgM_DENV",
"_1_resultado_teste_rapido_IgG_DENV",
"_1_teste_rapido_gravidez",
"_1_resultado_teste_rapido_gravidez",
"_1_TCLE_Biobanco",
"_1_uso_futuro_biobanco",
"_1_CHIKV_PCR _Request",
"_1_ZIKV_PCR_Request",
"_1_DENV_PCR _Request",
"_1_HCGQT_request",
"_1_TGO_request",
"_1_TGP_request",
"_1_no_collection_reason",
"_1_reason_no_collection",
"_1_visita_agendada",
"_2_visita_agendada",
"_3_visita_agendada",
"_4_visita_agendada",
"_1_visita",
"_2_visita",
"_3_visita",
"_4_visita",
"_1_coleta_sangue_retorno_DENV",
"_2_coleta_sangue_retorno_DENV",
"_1_coleta_sangue_retorno_CHIKV",
"_2_coleta_sangue_retorno_CHIKV",
"_3_coleta_sangue_retorno_CHIKV",
"_1_coleta _seg_gest_15d",
"_1_coleta_sangue_retorno_gestantes",
"_1_qex_denv_seg_7",
"_1_qex_seg_denv_14",
"_1_qex_seg_chikv_21",
"_1_qex_seg_chikv_90",
"_1_qex_seg_chikv_90_faltosos",
"_1_qex_seguimento_gestante",
"_1_encaminhamento_ultrassom",
"_1_qex_ultrassom_chikv",
"_2_qex_ultrassom_chikv",
"_1_ultrassom_imagem_chikv",
"_1_qex_ultrasson_gestantes",
"_1_ultrassom_imagem_gestantes",
"_1_data_ultrassom_imagem.1",
"_1_imagem_ultrassom_mal_formacao",
"_1_qex_nasc_gestantes",
"_1_seguimento_incompleto",
"_1_motivo_seguimento_incompleto",
"_1_outros_motivo_seguimento_incompleto",
"_1_dados_obito",
"_1_causa_basica_obito",
"_1_causa_basica_CID10",
"_1_qex_perda_seg_zikv",
"_1_inelegivel",
"_1_Elegivel",
"_1_Gestante",
"_1_agend_visitas",
"_1_Result_POS_denv",
"_1_Result_POS_chikv",
"_1_Result_POS_zikv",
"_1_Result_NEG",
"_1_PCR_CHIKV_Positive_Result",
"_1_PCR_CHIKV_Result",
"_1_PCR_CHIKV_Negative_Result",
"_1_PCR_ZIKV_Positive_Result",
"_1_PCR_ZIKV_Result",
"_1_PCR_ZIKV_Negative_Result",
"_1_PCR_DENV_Positive_Result",
"_1_PCR_DENV_Result",
"_1_PCR_DENV_Negative_Result",
"_1_falta_visita_agendada",
"_1_qex_triagem_finalizado",
"_1_qex_inclusao-finalizado",
"_1_prazo_ultrapassado_denv7",
"_1_prazo_ultrapassado_denv14",
"_1_prazo_ultrapassado_chikv21",
"_1_prazo_ultrapassado_chikv90",
"_1_prazo_ultrapassado_gestante",
"_1_qex_denv_d7_finalizado",
"_1_qex_denv_d14_finalizado",
"_1_qex_chikv_d21_finalizado",
"_1_qex_chikv_d90_finalizado",
"_1_qex_seg_gest_finalizado",
"_1_qex_nasc_gest_finalizado",
"_1_denv_retornos_completos",
"_1_chikv_retornos_completos",
"_1_ZIKV_Retornos_completos",
"_1_DENV_SA_Retornos_Completos",
"_1_due_scheduled_period",
"_1_TGO_Result",
"_1_TGP_Result",
"_1_HCGQT_Result",
"criterio_exclusao_30dias_medicamento",
"criterio_exclusao_doenca_coagulacao",
"criterio_exclusao_doenca_grave",
"criterio_exclusao_imunidade",
"criterio_exclusao_usa_anticoagulante",
"gestante_7D",
"gestante_14D",
"numero_pessoas_nucleo_familiar",
"febre_7dias",
"exantema_onde_outro",
"qual_doenca_das_articulacoes",
"qual_doenca_das_articulacoes_articulacoes_afetadas",
"qual_doenca_das_articulacoes_outra",
"sinais_alarme"]

for L in lista_drop_2: 
  base_unificada5_filtrada.drop(L, axis=1, inplace=True)
base_unificada5_filtrada.to_csv('base_sem_encode.csv')


corelacao=base_unificada5_filtrada.corr(method ='spearman')['fl_severidade']
corelacao.to_csv('correlation_dados_tarq.csv')

lista_varia=var_drop(base_unificada5_filtrada,variaveis_dummy_2)
len(lista_varia)
for k in lista_varia:
    base_unificada5_filtrada.drop(k,axis=1,inplace=True)


for k in ['laboratorio_hemograma_hematocrito','laboratorio_hemograma_hemoglobina']:
    base_unificada5_filtrada[k]=base_unificada5_filtrada[k].fillna('0,0')

for k in range(len(base_unificada5_filtrada['laboratorio_hemograma_hemoglobina'])):
  base_unificada5_filtrada['laboratorio_hemograma_hemoglobina'][k]=float(base_unificada5_filtrada['laboratorio_hemograma_hemoglobina'][k].replace('.','').replace(',','.'))

for k in range(len(base_unificada5_filtrada['laboratorio_hemograma_hematocrito'])):
  base_unificada5_filtrada['laboratorio_hemograma_hematocrito'][k]=float(base_unificada5_filtrada['laboratorio_hemograma_hematocrito'][k].replace('.','').replace(',','.'))

base_unificada5_filtrada['laboratorio_hemograma_hemoglobina']=base_unificada5_filtrada['laboratorio_hemograma_hemoglobina'].astype('float')
base_unificada5_filtrada['laboratorio_hemograma_hematocrito']=base_unificada5_filtrada['laboratorio_hemograma_hematocrito'].astype('float')

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
amostra_paci=base_unificada5_filtrada[base_unificada5_filtrada['Subject_ID'].isin(ids['Subject_ID'])].reset_index()

amostra_paci_21=amostra_paci.copy()
amostra_paci_31=amostra_paci.copy()
amostra_paci.drop('Subject_ID', axis=1, inplace=True)
amostra_paci.drop('level_0', axis=1, inplace=True)

amostra_paci=category2(amostra_paci,C)
len(amostra_paci.columns)
amostra_paci=amostra_paci.fillna(0.0) 

x_train,x_test,y_train,y_test=train_test_split(amostra_paci,amostra_paci['fl_severidade'],test_size=0.5,random_state=0)
test=y_test.sum()

while test<6 or test>9:
    x_train,x_test,y_train,y_test=train_test_split(amostra_paci,amostra_paci['fl_severidade'],test_size=0.5)
    test=y_test.sum()

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

pickle_lista = open("vars_const.pickle","wb")
pickle.dump(vars_const, pickle_lista)
pickle_lista.close()

amostra_paci2=amostra_paci.copy()
amostra_paci3=amostra_paci.copy()

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
y_train=y_train.fillna(0) 
y_test=y_test.fillna(0) 
print('Modelo Logistica')
print(logistic(x_train,x_test,y_train,y_test,amostra_paci,fl,amostra_paci2))
print('')
print('Modelo Bayes Complement')
print(complement_bayes(x_train,x_test,y_train,y_test,amostra_paci,fl,amostra_paci3))