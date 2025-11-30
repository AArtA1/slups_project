#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from utils import ols_cir, plot_history


# In[2]:


file_name = '../data/SOFR.csv'


# In[20]:


df = pd.read_csv(file_name)


# In[21]:


df['Date'] = pd.to_datetime(df['observation_date'])
df = df.sort_values('Date').set_index('Date')


# In[22]:


df = df['2019-01-01':'2021-12-10']


# In[23]:


df['r'] = df['SOFR'].values / 100

r = df['r'].dropna().values

dt = 1 / 252

print(f"✅ Данные загружены! Всего точек: {len(r)}")
print(f"Пример ставки (доли): {r[:5]}")


# In[24]:


plot_history(df['r'], "Историческая ставка USD (SOFR)", "Ставка, %")


# In[30]:


kappa, theta, sigma = ols_cir(r, dt)


# In[31]:


print("\n=== 🎯 РЕЗУЛЬТАТЫ ДЛЯ SOFR (USD) ===")
print(f"Kappa: {kappa:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Sigma: {sigma:.4f}")

