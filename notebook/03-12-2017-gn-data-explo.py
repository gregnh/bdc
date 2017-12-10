
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib notebook')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
pd.options.display.max_columns = 35


# In[2]:

data_path = '../data/raw/'


# In[3]:

train = pd.read_csv(data_path + "train_30.csv", header=0, delimiter=";",decimal=',',
                    parse_dates=['date'], index_col='date')


# In[4]:

train.head(5)


# In[5]:

train.tail(1)


# In[6]:

train.info()


# ddH10_rose4, mois object type => category ?  
# There are missing values

# In[7]:

train.describe(include='all')


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

# # Rows with nan

# In[8]:

# Total missings values by variable
pd.DataFrame(train.isnull().sum()).T


# flir1SOL0 	fllat1SOL0 	flsen1SOL0 	flvis1SOL0 iwcSOL0 	nbSOL0_HMoy ntSOL0_HMoy rr1SOL0 tpwHPA850 : 7 missings  
# rrH20 : 119 missings  
# clwcH20 ddH10_rose4 ffH10 huH2 tH2 	tH2_VGrad_2.100 tH2_XGrad tH2_YGrad ux1H10 vx1H10 : 126 missings  
# pMER0 : 133  
# nH20 : 336  
# capeinsSOL0 : 616 

# In[9]:

for i in np.where(train.isnull().sum() == 7)[0]:
    print(np.where(np.asanyarray(pd.isnull(train.iloc[:,i]))))


# Same location for these missings vals

# In[10]:

# Rows with missing val
np.where(train.isnull().sum(axis=1).values > 0)[0]


# In[11]:

# Coordinate of missing values
np.where(np.asanyarray(pd.isnull(train)))


# Lets have a look to 1 row

# In[12]:

pd.DataFrame(train.iloc[539+(7*11),:]).T


# ### Lets see if one can interpolate the missing values

# In[13]:

for i in range(1,8):
    print(train.index[539-7*i], 
          ' - capeinsSOL0 val : {} for station {}'.format(train.capeinsSOL0[539-7*i], train.insee[539-7*i]))
print('interpolate : {}'.format(train.capeinsSOL0.interpolate()[539]))


# In[14]:

for i in range(1,25):
    print(train.capeinsSOL0[539+7*i])


# ### Difficile d'interpoler (ici pour capeinsSOL0)

# In[15]:

for i in range(1,5):
    print(train.index[4627-7*i], 
          ' - flir1SOL0 val : {} for station {}'.format(train.flir1SOL0[4627-7*i], train.insee[4627-7*i]))
print('interpolate : {}'.format(train.flir1SOL0.interpolate()[4627]))
for i in range(1,5):
    print(train.flir1SOL0[4627+7*i])


# ### Ici ca marche. A explorer

# ### Replace na val en utilisant le groupby. Faire des stats par city
# 
# https://machinelearningmastery.com/time-series-data-visualization-with-python/

# In[16]:

# train.insee = train.insee.astype('category')
# groupby_city = train.groupby('insee')
# groupby_city.capeinsSOL0.apply(pd.Series.interpolate)


# ## Missing values by file

# In[17]:

df_miss_val = pd.DataFrame(columns=train.columns)
for i in range(1,37):
    tmp =pd.read_csv(data_path + 'train_' + str(i) + '.csv', 
                     header=0, delimiter=";",decimal=',',
                     parse_dates=['date'], index_col='date')
    df_miss_val = pd.concat([df_miss_val, pd.DataFrame(tmp.isnull().sum()).T], axis=0 )
df_miss_val


# In[ ]:




# # Link between temp and ech

# ## Data processing
# Stack data

# In[57]:

from os import listdir

temp_ech = None
cols = ['date', 'insee', 'tH2_obs', 'tH2', 'ech']
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
temp_ech.head(8)


# ## Analysis

# In[59]:

print(max(temp_ech.index)-min(temp_ech.index))
temp_ech['2014-01-01']


# On peut voir que quand temp à n+1 with ech=1 == temp à n with ech=25 but not true with other cols

# Dealing with missing val :  
# https://gallery.cortanaintelligence.com/Experiment/Methods-for-handling-missing-values-1

# In[ ]:



