
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib notebook')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
pd.options.display.max_columns = 35

data_path = '../data/raw/'


# In[2]:

train = pd.read_csv(data_path + "train_1.csv", header=0, delimiter=";",decimal=',',
                    parse_dates=['date'], index_col='date')


# In[3]:

train.head(5)


# In[4]:

train.tail(1)


# In[5]:

train.info()


# ddH10_rose4, mois object type => category ?  
# There are missing values

# In[6]:

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

# In[7]:

# Total missings values by variable
pd.DataFrame(train.isnull().sum()).T


# In[8]:

for i in np.where(train.isnull().sum() == 126)[0]:
    print(np.where(np.asanyarray(pd.isnull(train.iloc[:,i]))))


# Same row location for missing vals for these variables

# In[9]:

# Rows with missing val
np.where(train.isnull().sum(axis=1).values > 0)[0]


# In[10]:

# Coordinate of missing values
np.where(np.asanyarray(pd.isnull(train)))


# Lets have a look to 1 row

# In[49]:

pd.DataFrame(train.iloc[2653-7*3,:]).T


# ## Missing values by file

# In[3]:

df_miss_val = pd.DataFrame(columns=train.columns).drop(['insee', 'tH2_obs', 'ech', 'mois'], axis=1)

for i in range(1,37):
    tmp =pd.read_csv(data_path + 'train_' + str(i) + '.csv', 
                     header=0, delimiter=";",decimal=',',
                     parse_dates=['date'], index_col='date').drop(['insee', 'tH2_obs', 'ech', 'mois'], axis=1)
    missing_val_r, missing_val_c = np.where(np.asanyarray(pd.isnull(tmp)))
    tmp_row_df = [[] for _ in range(len(tmp.columns.values))]
    for row, col in zip(missing_val_r, missing_val_c):
        tmp_row_df[col].append(row)
    tmp_row_df = np.array(tmp_row_df).reshape([1,-1])
    # total number of missing val per col
#     df_miss_val = pd.concat([df_miss_val, pd.DataFrame(tmp.isnull().sum()).T], axis=0)
    df_miss_val = pd.concat([df_miss_val, pd.DataFrame(tmp_row_df, columns=tmp.columns)], axis=0, ignore_index=True)
df_miss_val


# In[4]:

np.corrcoef(map(len, df_miss_val.loc[0])), df_miss_val.loc[0]


# In[5]:

## For sum sur chaque vector: good method or not to see correlation
for i in df_miss_val.columns.values:
    print(sum(df_miss_val.loc[0,i]), '--', i)


# Pb : dont allow to keep the coordinate

# In[25]:

dict_corr_matrix_of_cols_with_matching_missing_vals = dict()
for file_number in range(len(df_miss_val)): #replace .loc[0] with .loc[file_number]
    dict_corr_matrix_of_cols_with_matching_missing_vals[file_number+1] = dict()
    len_missing_val_per_cols = map(len, df_miss_val.loc[file_number])
#     print('Data missing for the variables:', np.array(len_vect_cols))
    
    for len_vect in set(len_missing_val_per_cols):
        dict_corr_matrix_of_cols_with_matching_missing_vals[file_number+1][len_vect] = dict()
        if len_vect > 0:
#             print('Variables of missing values:', len_vect)
            cols_matching_len_vect = np.where(np.array(len_missing_val_per_cols) == len_vect)[0]
            if len(cols_matching_len_vect) > 1:
                dict_corr_matrix_of_cols_with_matching_missing_vals[file_number+1][len_vect]['matching_cols'] = cols_matching_len_vect
                corr = np.corrcoef(list(df_miss_val.iloc[file_number,cols_matching_len_vect]))
                dict_corr_matrix_of_cols_with_matching_missing_vals[file_number+1][len_vect]['corr_mat'] = corr


# In[24]:

dict_corr_matrix_of_cols_with_matching_missing_vals


# vectors of similar length have the same coordinates

# Dealing with missing val :  
# https://gallery.cortanaintelligence.com/Experiment/Methods-for-handling-missing-values-1

# In[ ]:



