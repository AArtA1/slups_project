#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


file_name = 'data/usdrub_data.csv'


# In[ ]:


df = pd.read_csv(file_name)


# In[ ]:


col_name = 'RUB=X' if 'RUB=X' in df.columns else df.columns[1]


# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').set_index('Date')


# In[ ]:


df = df['2019-01-01':'2021-12-10']


# In[ ]:


prices = df[col_name]
log_returns = np.log(prices / prices.shift(1)).dropna()


# In[ ]:


sigma_fx = log_returns.std() * np.sqrt(252)


# In[ ]:


print("\n=== 🎯 РЕЗУЛЬТАТ ДЛЯ КУРСА (USDRUB) ===")
print(f"Sigma FX: {sigma_fx:.4f} ({sigma_fx:.2%})")


# In[ ]:




