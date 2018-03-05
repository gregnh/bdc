
# coding: utf-8

# idée de Grégoire

# In[13]:

import pandas as pd
import numpy as np
import sys
sys.path.append('../')
data_path = '../data/raw/'


# ## Data processing
# Stack data

# In[16]:

from os import listdir

cols = ['date', 'insee', 'tH2_obs', 'tH2', 'ech']
temp_ech = None
for file_ in listdir(data_path):
    if temp_ech is None:
        temp_ech = pd.read_csv(data_path + file_, header=0, 
                               delimiter=";",decimal=',',parse_dates=['date'], index_col='date', usecols=cols)
    else:
        temp_ech = pd.concat([temp_ech, 
                              pd.read_csv(data_path + file_, header=0, delimiter=";",decimal=',', 
                                          parse_dates=['date'], index_col='date', usecols=cols
                                         )])
# train.to_csv(data_path + "train.csv", sep=';', index=False)
temp_ech.sort_values(['insee', 'ech'], ascending=[True, True], inplace=True)


# In[69]:

temp_ech[temp_ech['insee'] == 6088001]['2014-01-01':'2014-01-02'].head(55)


# ## Analysis

# In[54]:

print(max(temp_ech.index)-min(temp_ech.index))
temp_ech['2014-01-01']


# On peut voir que quand temp à n+1 with ech=1 == temp à n with ech=25 but not true with other cols

# In[38]:

from os import listdir

cols = ['date', 'insee', 'tH2_obs', 'tH2', 'ech']
df = pd.DataFrame(columns=cols+['tH2_ech_n'])
for file_number in range(1,36+1):
    tmp = pd.read_csv(data_path + 'train_' + str(file_number) + '.csv', header=0, 
                               delimiter=";",decimal=',',parse_dates=['date'], index_col='date', usecols=cols)
    if df.ech > 1:
        
    tmp['tH2_ech_n'] = pd.Series(np.empty((3,1)).fill(np.nan), index= tmp.index)
    
#     if df.ech
#     df = pd.concat([df, ])
# train.to_csv(data_path + "train.csv", sep=';', index=False)

# df.sort_values(['insee', 'ech'], ascending=[True, True], inplace=True)


# In[51]:

a = tmp.index[8] - tmp.index[2]
a.days


# In[84]:

(temp_ech.index[0]).day


# In[ ]:



