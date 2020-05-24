# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:01:17 2020

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

    
base_unificada5.drop('sinais_alarme_qual', axis=1, inplace=True)
base_unificada5.drop('sinais_de_alarme_7D', axis=1, inplace=True)
base_unificada5.drop('sinais_de_alarme_14D', axis=1, inplace=True)
base_unificada5.drop('sintoma_febre_7dias', axis=1, inplace=True)
base_unificada5.drop('outros_sinais_alarme_quais', axis=1, inplace=True)

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
"qual_doenca_das_articulacoes_outra"]

for L in lista_drop_2: 
  base_unificada5_filtrada.drop(L, axis=1, inplace=True)
base_unificada5_filtrada.to_csv('base_sem_encode.csv')

C=base_unificada5_filtrada.select_dtypes(include=['object'])
C.to_csv('strings.csv')
  
for K in C.columns:
    ce_target_leaf = ce.TargetEncoder(cols = [K])
    ce_target_leaf.fit(base_unificada5_filtrada[K], base_unificada5_filtrada['fl_severidade'])
    base_unificada5_filtrada[K] = ce_target_leaf.transform(base_unificada5_filtrada[K], base_unificada5_filtrada['fl_severidade'])
    

base_unificada5_filtrada=base_unificada5_filtrada.fillna(0.0)   
base_unificada5_filtrada.to_csv('base_total.csv')

#################################################################
#LIMPEZA DE VARIAVEIS CONSTANTES
##################################################################
vl_limpa_const=0.1 # variaveis com 70% dos campos repetidos
limpa_const = VarianceThreshold(threshold= vl_limpa_const)
limpa_const.fit(base_unificada5_filtrada)
vars_const = [v for v in base_unificada5_filtrada.columns if v not in  base_unificada5_filtrada.columns[limpa_const.get_support()]]
qt_var = len([v for v in base_unificada5_filtrada.columns if v not in  base_unificada5_filtrada.columns[limpa_const.get_support()]])
 
print('Existem {} variaveis constantes com limite de {}'.format(qt_var,vl_limpa_const))
print('Variaveis constantes com limite de {}'.format(vl_limpa_const))
print(vars_const)
d = {'vars_const': vars_const}
df = pd.DataFrame(data=d)
df.to_csv('vars_const.csv') 
 
#base_unificada6 = base_unificada5.drop(vars_const, axis = 1)
pickle_lista = open("vars_const.pickle","wb")
pickle.dump(vars_const, pickle_lista)
pickle_lista.close()

#Selecionando amostra para modelagem após os tratamentos 
#Realizados em todos os dados
# E separando em treino e teste

base_para_score=base_unificada5_filtrada.copy()
#amostra_paci =base_unificada5_filtrada.sample(frac=0.5, replace=False)
amostra_paci=base_unificada5_filtrada[base_unificada5_filtrada['Subject_ID'].isin(ids['Subject_ID'])].reset_index()

amostra_paci_2=amostra_paci.copy()
amostra_paci_3=amostra_paci_2.copy()

amostra_paci.drop('Subject_ID', axis=1, inplace=True)
amostra_paci.drop('index', axis=1, inplace=True)
amostra_paci.drop('level_0', axis=1, inplace=True)

corelacao=base_unificada5_filtrada.corr(method ='spearman')
corelacao.to_csv('correlation_dados.csv')

X_train, X_test = train_test_split(amostra_paci, test_size=0.50)
train=X_train['fl_severidade'].sum()
test=X_test['fl_severidade'].sum()

while 6<test<8:
    X_train, X_test = train_test_split(amostra_paci, test_size=0.50)
    train=X_train['fl_severidade'].sum()
    test=X_test['fl_severidade'].sum()

    
#Parametros para aplicação da modelagem (random forest)
#numero de amostras para uso no cross_validation da seleção de variaveis
qt_cv = 4
#numero de amostras para uso no cross_validation do desenvolvimento do modelo
qt_cv_2 = 5

#parametros para random forest sendo eles
#quantidade de arvores, profundidade das raizes e jobs
#100,125,150,200,250,300,325,350,375,
#400,450,475,500,525,550,575,600
#175
tuned_parameters = [{'n_estimators': [300,325,350,375,400,425,450],'max_depth': [4,5,6,7,8,9],'n_jobs':[3,4,5,6]}]

#separando variavel targuet do dataframe para realizar a modelagem
#tanto para teste quanto para treinamento
y_train=X_train['fl_severidade'].copy()
X_train.to_csv('base_train.csv')
del X_train['fl_severidade']
X_train=X_train.astype(float)


y_test=X_test['fl_severidade'].copy()
del X_test['fl_severidade']
X_test=X_test

##################################################################
#VARIAVEIS CORRELACIONADAS
##################################################################

corrmat = amostra_paci.corr(method ='spearman')
corrmat.to_csv('correlation_train_test.csv')

#Aplicação das tecnicas de modelagem apos o tratamento dos dados
modelo2= GridSearchCV(RandomForestClassifier(), tuned_parameters, cv= qt_cv, scoring='roc_auc')
modelo2.fit(X_train, y_train)

print('Melhores parametros')
print(modelo2.best_params_)
print(modelo2.best_estimator_)
########
########
  
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

probs_test = np.round(modelo2.predict_proba(X_test.values)[:,1], 7)
probs_train = np.round(modelo2.predict_proba(X_train.values)[:,1], 7)

D_test=pd.DataFrame(y_test[:])
D_test['score']=probs_test

D_train=pd.DataFrame(y_train[:])
D_train['score']=probs_train

p_c_test=D_test.score[D_test['fl_severidade']==1]
p_n_test=D_test.score[D_test['fl_severidade']==0]
p_c_train=D_train.score[D_train['fl_severidade']==1]
p_n_train=D_train.score[D_train['fl_severidade']==0]


fpr_test, tpr_test, thresholds = roc_curve(y_test, probs_test)
fpr_train, tpr_train, thresholds = roc_curve(y_train, probs_train)

print('KS de Validação: {0:.2f}% e AUC: {1:.2f}%'.format(stats.ks_2samp(p_c_test,p_n_test)[0] * 100.0, auc(fpr_test, tpr_test) * 100))
print('KS de Desenvolvimento: {0:.2f}% e AUC: {1:.2f}%'.format(stats.ks_2samp(p_c_train,p_n_train)[0] * 100.0, auc(fpr_train, tpr_train) * 100))

feature_importances = pd.DataFrame(modelo2.best_estimator_.feature_importances_, index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
feat_imp_all = feature_importances.sort_values(ascending=True, by='importance')
graf_all = feat_imp_all.plot(kind = 'barh', color = 'darkred', title='Variaveis', figsize=(15,60), grid = False )

feature_importances.to_csv("poder_variaveis_total.csv")
graf_all.figure.savefig('poder_variaveis_total_grafico.png')

#Salvando modelo 
Model_file = 'modelo_random.pkl'
pickle.dump(modelo2, open(Model_file, 'wb'))

#pickle.dump(modelo2,'modelo_random.pickle')
#Reprocessamento base inteira e aplico predict_proba na base inteira
#Determino os 4 grupos mais propensos e verifico se quem está lá condiz com 
#a caracteristica do tipo (severidade) da doença que esperamos


#Remove as variaveis que sabemos não ser uteis para o modelo, mas dessa vez
#iremos manter o id do panciente para realizar as quebras dos 4 grupos
variaveis_id2=[]
for k in amostra_paci_2.columns:
    if k not in X_train.columns:
        variaveis_id2.append(k)
        
base_ids2=amostra_paci_2[variaveis_id2]  
   
for L in variaveis_id2: 
  amostra_paci_2.drop(L, axis=1, inplace=True)

   
from math import log
vetor_probabilidades_amostra=[]
for k in range(len(amostra_paci_2)):
  #vetor_probabilidades_amostra.append(int(round(500+(log(1/(1-np.round(modelo2.predict_proba(amostra_paci_2[0:].values)[:,1], 7)[k]))*20/log(2)),0)))
   vetor_probabilidades_amostra.append(np.round(modelo2.predict_proba(amostra_paci_2[0:].values)[:,1], 7)[k])   
  
ve=[]
for k in vetor_probabilidades_amostra:
 ve.append(k)

amostra_paci_2['Subject_ID']=base_ids2['Subject_ID']
amostra_paci_2['fl_severidade']=base_ids2['fl_severidade']

df_vetor={'score':ve}
df_vetor_final=pd.DataFrame(df_vetor,columns=['score'])
print(df_vetor_final)
print()
print(base_unificada5_filtrada)
print()
amostra_paci_2['score']=df_vetor_final['score']
amostra_paci_2=amostra_paci_2.sort_values('score',ascending=False)
amostra_paci_2.to_csv("base_scorada_amostra.csv") 
import seaborn as sns 
sns.boxplot(x=amostra_paci_2['score'])
plt.savefig('boxplot_random.png')
plt.plot(amostra_paci_2['score'],'*')
plt.savefig('distribuicao_random.png')

###################  GRADIENT BOOSTING 
############################################
########
from sklearn.ensemble import GradientBoostingClassifier
tuned_parameters1= {
    "loss":["deviance","exponential"],
    "learning_rate": [0.05,0.075],
    "min_samples_split": np.linspace(0.1, 0.5, 6),
    "min_samples_leaf": np.linspace(0.1, 0.5, 6),
    "max_depth":[4,5,6],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse","mae"],
    "subsample":[0.5,0.6,0.7],
    "n_estimators":[400,450,500,550]
    }

modelo3= GridSearchCV(GradientBoostingClassifier(), tuned_parameters1, cv= qt_cv, n_jobs=-1, scoring='roc_auc')
modelo3.fit(X_train, y_train)

print(modelo3.best_params_)
probs_test = np.round(modelo3.predict_proba(X_test.values)[:,1], 7)
probs_train = np.round(modelo3.predict_proba(X_train.values)[:,1], 7)

D_test=pd.DataFrame(y_test[:])
D_test['score']=probs_test

D_train=pd.DataFrame(y_train[:])
D_train['score']=probs_train

p_c_test=D_test.score[D_test['fl_severidade']==1]
p_n_test=D_test.score[D_test['fl_severidade']==0]
p_c_train=D_train.score[D_train['fl_severidade']==1]
p_n_train=D_train.score[D_train['fl_severidade']==0]

fpr_test, tpr_test, thresholds = roc_curve(y_test, probs_test)
fpr_train, tpr_train, thresholds = roc_curve(y_train, probs_train)

print('KS de Validação: {0:.2f}% e AUC: {1:.2f}%'.format(stats.ks_2samp(p_c_test,p_n_test)[0] * 100.0, auc(fpr_test, tpr_test) * 100))
print('KS de Desenvolvimento: {0:.2f}% e AUC: {1:.2f}%'.format(stats.ks_2samp(p_c_train,p_n_train)[0] * 100.0, auc(fpr_train, tpr_train) * 100))


feature_importances = pd.DataFrame(modelo3.best_estimator_.feature_importances_,X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances.count())
print(modelo3.best_estimator_.feature_importances_)


feat_imp_all = feature_importances.sort_values(ascending=True, by='importance')
graf_all = feat_imp_all.plot(kind = 'barh', color = 'darkred', title='Variaveis', figsize=(15,60), grid = False )

feature_importances.to_csv("poder_variaveis_total_gradient.csv")
graf_all.figure.savefig('poder_variaveis_total_grafico_gradient.png')


variaveis_id3=[]
for k in amostra_paci_3.columns:
    if k not in X_train.columns:
        variaveis_id3.append(k)
        
base_ids3=amostra_paci_3[variaveis_id3]  
   
for L in variaveis_id3: 
  amostra_paci_3.drop(L, axis=1, inplace=True)

   
from math import log
vetor_probabilidades_amostra=[]
for k in range(len(amostra_paci_3)):
   #vetor_probabilidades_amostra.append(int(round(500+(log(1/(1-np.round(modelo2.predict_proba(amostra_paci_2[0:].values)[:,1], 7)[k]))*20/log(2)),0)))
   vetor_probabilidades_amostra.append(np.round(modelo3.predict_proba(amostra_paci_3[0:].values)[:,1], 7)[k])   
  
ve=[]
for k in vetor_probabilidades_amostra:
 ve.append(k)

amostra_paci_3['Subject_ID']=base_ids3['Subject_ID']
amostra_paci_3['fl_severidade']=base_ids3['fl_severidade']

df_vetor={'score':ve}
df_vetor_final=pd.DataFrame(df_vetor,columns=['score'])
print(df_vetor_final)
print()
amostra_paci_3['score']=df_vetor_final['score']
amostra_paci_3=amostra_paci_3.sort_values('score',ascending=False)
amostra_paci_3.to_csv("base_scorada_amostra_gradiente.csv") 

import seaborn as sns 
plt.plot(amostra_paci_3['score'],'+')
plt.savefig('distribuicao_gradiente.png')
sns.boxplot(x=amostra_paci_3['score'])
plt.savefig('boxplot_gradient.png')