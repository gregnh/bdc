
# coding: utf-8

# # TODO
# - dataviz

# In[25]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sb
import time, sys

sys.path.append('../')

pd.options.display.max_columns = 50


# In[2]:

data_path = '../data/'
train = pd.read_csv(data_path + "raw/train_30.csv", header=0, delimiter=";",decimal=',',
                    parse_dates=['date'], index_col='date')


# **tH2_obs** :	Observation de la température à 2 mètres in situ- au point station (prédictant)  
# **capeinsSOL0**: 	Energie potentielle convective  
# **ciwcH20**: 	Fraction de glace nuageuse à 20 mètres  
# **clwcH20**: 	Fraction d'eau nuageuse à 20 mètres  
# **ddH10_rose4**: 	Direction du vent à 10 mètres en rose4  
# **ffH10**: 	Force du vent à 10 mètres en m/s  
# **flir1SOL0**: 	Flux Infra-rouge en J/m2  
# **fllat1SOL0**: 	Flux de chaleur latente en J/m2  
# **flsen1SOL0**: 	Flux de chaleur sensible en J/m2  
# **flvis1SOL0**: 	Flux visible en J/m2  
# **hcoulimSOL0**: 	Hauteur de la couche limite en mètres  
# **huH2**: 	Humidité 2mètres en %  
# **iwcSOL0**: 	Réservoir neige kg/m2 (équivalent en eau liquide des chutes de neige)  
# **nbSOL0_HMoy**: 	Nébulosité basse (moyenne sur les 6 points de grille autour de la station) (fraction en octat du ciel occulté)  
# **nH20**: 	Fraction nuageuse à 20 mètres  
# **ntSOL0_HMoy**: 	Nébulosité totale (moyenne sur les 6 points de grille autour de la station)  
# **pMER0**: 	Pression au niveau de la mer  
# **rr1SOL0**: 	Précipitation horaire au niveau du sol  
# **rrH20**: 	Précipitation horaire à 20 mètres  
# **tH2**: 	Température à 2 mètres du modèle AROME  
# **tH2_VGrad_2.100**: 	Gradient vertical de température entre 2 mètres et 100 mètres  
# **tH2_XGrad**: 	Gradient zonal de température à 2 mètres  
# **tH2_YGrad**: 	Gradient méridien de température à 2 mètres  
# **tpwHPA850**: 	Température potentielle au niveau 850 hPa  
# **ux1H10**: 	Rafale 1 minute du vent à 10 mètres composante zonale  
# **vapcSOL0**: 	Colonne de vapeur d'eau  
# **vx1H10**: 	Rafale 1 minute du vent à 10 mètres composante verticale  
# **ech** : 	Echéance de validité = date   

# # Converting type

# In[3]:

train.insee = train.insee.astype('str')


# # Univariate analysis

# In[28]:

plt.figure()
train.plot(subplots=True, figsize=(20,60), grid=True)
plt.show()


# Thats a mess, lets analyze it per week

# In[5]:

train.columns.values


# In[6]:

var_used = ['capeinsSOL0', 'ciwcH20', 'clwcH20',
       'ffH10', 'flir1SOL0', 'fllat1SOL0', 'flsen1SOL0',
       'flvis1SOL0', 'hcoulimSOL0', 'huH2', 'iwcSOL0', 'nbSOL0_HMoy',
       'nH20', 'ntSOL0_HMoy', 'pMER0', 'rr1SOL0', 'rrH20', 'tH2',
       'tH2_VGrad_2.100', 'tH2_XGrad', 'tH2_YGrad', 'tpwHPA850', 'ux1H10',
       'vapcSOL0', 'vx1H10']


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


# In[30]:

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


# In[ ]:




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

