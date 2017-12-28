
# coding: utf-8

# # TODO
# - dataviz

# In[1]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time, sys

sys.path.append('../')

pd.options.display.max_columns = 50


# In[2]:

data_path = '../data/'
train = pd.read_csv(data_path + "raw/train_1.csv", header=0, delimiter=";",decimal=',',
                    parse_dates=['date'], index_col='date')
train.dropna(how='any', inplace=True)


# In[3]:

train.columns.values


# In[4]:

used_var = ['ffH10', 'flir1SOL0', 'fllat1SOL0', 'flsen1SOL0',
       'flvis1SOL0', 'hcoulimSOL0', 'huH2', 'iwcSOL0', 'nbSOL0_HMoy',
       'nH20', 'ntSOL0_HMoy', 'pMER0', 'rr1SOL0', 'rrH20', 'tH2',
       'tH2_VGrad_2.100', 'tH2_XGrad', 'tH2_YGrad', 'tpwHPA850', 'ux1H10',
       'vapcSOL0', 'vx1H10']


# # Univariate analysis

# In[5]:

gb_cities = train.groupby('insee')


# ## Density

# In[6]:

for key, val in gb_cities:
    print(key)


# In[7]:

gb_cities['capeinsSOL0'].get_group(31069001)


# In[19]:

fig, ax = plt.subplots()
for var in used_var:
    g = sb.FacetGrid(train, hue='insee', size=3, aspect=4,
                     palette=sb.color_palette("Set1", n_colors=7, desat=.5))
    g.map(sb.distplot, var, hist=False)
    sb.plt.legend()


# For most variables, 6088001 does not have exactly the same density

# ## Plot variables

# In[28]:

plt.figure()
train.plot(subplots=True, figsize=(20,60), grid=True)
plt.show()


# Thats a mess, lets analyze it per week  
# Data semble stationaire

# ## Weekly analysis

# In[13]:

get_ipython().magic(u'matplotlib notebook')
weekly = pd.DataFrame()
for var in var_used:
    weekly = train[var].resample('W').apply(['mean', np.min, np.max, 'std'])
    weekly.plot(subplots=True, title=var)


# In[54]:

weekly_std = pd.DataFrame()
for var in var_used:
    tmp = train[var].resample('W').apply([np.std])
    tmp.rename(index=str, columns={"std": var}, inplace=True)
    weekly_std = pd.concat([weekly_std, tmp], axis=1)
weekly_std.index = pd.DatetimeIndex(weekly_std.index).normalize() # dropping time
weekly_std


# ## Analysis per city

# ### Temp

# In[29]:

groupby_city = train.groupby('insee')


# In[ ]:

figure = plt.figure(num=None, figsize=(25,45), dpi=35, facecolor='w', edgecolor='k')
nb_row = 7
nb_col = 1
i = 1
for city, vars_ in groupby_city:
    plt.subplot(nb_row, nb_col, i)
    plt.plot(vars_.index, vars_.tH2_obs, '--', label=city)
    i +=1
    plt.title(city)
    plt.grid()
plt.show()


# # Draft

# **Tout mettre dans 1 fichier**

# In[3]:

# from os import listdir
# train_path = data_path + 'train/'
# train = None
# for file_ in listdir(train_path):
#     if 'train' in file_:
#         if train is None:
#             train = pd.read_csv(train_path + file_, header=0, delimiter=";", parse_dates=['date'])
#         else:
#             train = pd.concat([train, pd.read_csv(train_path + file_, header=0, delimiter=";", parse_dates=['date'])])
# train.to_csv(data_path + "train.csv", sep=';', index=False)


# In[ ]:

# pickle_file = open(file_path, 'wb')
#     pickle.dump(data, pickle_file, pickle.HIGHEST_PROTOCOL)
#     pickle_file.close()

