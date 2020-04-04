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

dados_analises=pd.read_csv('pt_BR_0_arbobios2-20200309041123_v2.csv',sep=';',error_bad_lines=False)
dados_analises2=pd.read_csv('questionario_pt_BR_0_CRF_01 Critério de elegibilidade-20200309061346_v2.csv',sep=';',error_bad_lines=False)
dados_analises3=pd.read_csv('questionario_pt_BR_0_CRF_05 Dengue D7-20200309061441_v2.csv',sep=';',error_bad_lines=False)
dados_analises4=pd.read_csv('questionario_pt_BR_0_CRF_06 Dengue D14-20200309061459_v2.csv',sep=';',error_bad_lines=False)

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

base_unificada5.to_csv('base.csv')

lista_trop=[
"Hospital",
"DOB",
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
"_1_HCGQT_Result_Created_Date"]


for L in lista_trop: 
  base_unificada5.drop(L, axis=1, inplace=True)
  
  
  
  

  
'''
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
limpa_const.fit(base_unificada5)
vars_const = [v for v in base_unificada5.columns if v not in  base_unificada5.columns[limpa_const.get_support()]]
qt_var = len([v for v in base_unificada5.columns if v not in  base_unificada5.columns[limpa_const.get_support()]])
 
print('Existem {} variaveis constantes com limite de {}'.format(qt_var,vl_limpa_const))
print('Variaveis constantes com limite de {}'.format(vl_limpa_const))
print(vars_const)
 
 
base_unificada6 = base_unificada5.drop(vars_const, axis = 1)
vars_const.to_pickle('vars_const.pickle')

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
'''



