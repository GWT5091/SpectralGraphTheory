#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('../data/20200524_L.csv',index_col=0)


# In[3]:


la, v = np.linalg.eig(df.values)


# In[4]:


for i in range(len(la)):
    print('第'+str(i)+'成分')
    print(la[i])
    print(v[i])


# In[12]:




