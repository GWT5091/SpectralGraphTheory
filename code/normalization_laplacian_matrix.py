#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import datetime as dt

def convert_nor_lap_mat(df:pd.DataFrame):
    size_taple = df.values.shape
    data_val = df.values
    matrix = np.zeros(size_taple)
    for i in range(size_taple[0]):
        for j in range(size_taple[0]):
            if i == j:
                matrix[i][j] = data_val[i][j]
            else:
                matrix[i][j] = (data_val[i][j] / np.sqrt(data_val[i][i] * data_val[j][j]))*(-1)
    return pd.DataFrame(index=df.index, columns=df.columns, data = matrix)

def today():
    tdatetime = dt.datetime.now()
    return tdatetime.strftime('%Y%m%d')

def output_csv(df:pd.DataFrame, name:str):
    df.to_csv('../output/'+today()+name+'.csv',encoding='utf-8-sig')


# In[22]:


path = '../data/'
data_name = '20200422_L.csv'
data_df = pd.read_csv(path+data_name,index_col=0)
N_data_df = convert_nor_lap_mat(data_df)
output_csv(N_data_df,'Nomalization_laplacian_matrix')

