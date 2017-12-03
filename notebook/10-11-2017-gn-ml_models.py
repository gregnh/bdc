
# coding: utf-8

# # TODO
# - feature engineering sur la datetime (add year, season, weeknumber)
# - LSTM w/ Keras
# - ARIMA & SARIMA Model

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd, numpy as np
import sys, os

import sklearn.preprocessing as pp
import sklearn.decomposition as decomposition
import sklearn.model_selection as ms
import sklearn.metrics as metrics

import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
# import xgboost
# import keras
sys.path.append('../')
# import lib.tools
pd.options.display.max_columns = 50


# In[2]:

raw_path = '../data/raw/'


# # First approach : prediction by removing Nan rows

# ## Data Preprocessing

# In[3]:

def season(df):
    df['month'] = df.index.month
    season_col = df['month']
    season_col = season_col.replace([12, 1, 2], 0)
    season_col = season_col.replace([3, 4, 5], 1)
    season_col = season_col.replace([6, 7, 8], 2)
    season_col = season_col.replace([9, 10, 11], 3)
    ohc = pd.DataFrame(np.eye(4)[season_col], columns=['winter', 'spring', 'summer', 'fall'], index=df.index)
#     for col in ohc:
#         ohc[col] = ohc[col].astype('category')
    return pd.concat([df, ohc], axis=1)


# In[4]:

# for file_ in os.listdir(train_path):
file_ = pd.read_csv(raw_path + 'train_1.csv', header=0, delimiter=';',decimal=',',
                    parse_dates=['date'], index_col='date')

file_.dropna(how='any', axis=0, inplace=True)
file_ = file_.rename(columns={'mois':'month'})

#Transform to dummy
dummies = ['ddH10_rose4', 'insee', 'month']
df_dummy = pd.get_dummies(file_, columns=dummies, prefix=dummies)
#     for col in df_dummy:
#         df_dummy[col] = df_dummy[col].astype('category')
file_.drop(['ddH10_rose4'], inplace=True, axis=1)

# Add temporal features
# file_['week'] = file_.index.week # WEEK AS CATEGORY ?
file_ = season(file_) # add season 
file_.drop(['month'], inplace=True, axis=1)

# Add lag operator (shift ce fait par ville)
groupby_cities = file_.groupby('insee')
shift = groupby_cities.tH2_obs.shift(1)
file_['tH2_obs_shift_1'] = shift

# file_.drop(['insee'], inplace=True, axis=1)


# In[5]:

file_.describe(include='all')


# In[6]:

file_.head(14)


# ## Factor Analysis

# In[97]:

# file_['2014-01'][]
# calculer la variation moyenne de temperature en de décembre à janvier pour estimer 2014-01-01


# In[98]:

file_.dropna(how='any', axis=0, inplace=True)
fa = decomposition.FactorAnalysis().fit(file_.drop('tH2_obs'))


# In[99]:

fa


# ## ML

# In[ ]:

seed = 1
results = []
names = []
predictions = [] #table of prediction for each model


# In[ ]:

models = []
models.append(('CART', tree.DecisionTreeRegressor(n_jobs=3)))
models.append(('RF', ensemble.RandomForestRegressor(n_jobs=3)))
models.append(('KNN', neighbors.KNeighborsRegressor(n_jobs=3)))
models.append(('SVM', svm.SVR(n_jobs=3)))
models.append(('GB', ensemble.GradientBoostingRegressor(n_jobs=3)))
# models.append(('NB'), ) # Naive Bayes for modeling uncertainty
# LSTM


# Apply forward chaining

# In[14]:

first_train = True
for i in range(1,36):
    train_file = train_path + 'train_' + str(i) + '.csv'
    test_file = train_path + 'train_' + str(i+1) + '.csv'
    
    x_test = pd.read_csv(test_file, header=0, delimiter=';',decimal=',',
                    parse_dates=['date'], index_col='date').drop('tH2_obs', axis=1)
    y_test = pd.read_csv(test_file, header=0, delimiter=';',decimal=',',
                    parse_dates=['date'], index_col='date', usecols=['tH2_obs'])
    train = pd.read_csv(train_file, header=0, delimiter=';',decimal=',',
                    parse_dates=['date'], index_col='date')
    
    
    if first_train is True:
        x_train = train.drop('tH2_obs', axis=1)
        y_train = train.pop('tH2_obs')
        first_train = False
    else:
        x_train = pd.concat([x_train, train.drop('tH2_obs', axis=1)])
        y_train = pd.concat([y_train, train.pop('tH2_obs')])
        
    for name, m in models: 
    pred = ms.cross_val_predict(m, dff[features], dff[label], cv=kfold, n_jobs=3)
    cv_res = ms.cross_val_score(m, dff[features], dff[label], cv=kfold, n_jobs=3, scoring = 'accuracy')
    m.fit
    
    predictions.append(pred)
    results.append(cv_res)
    names.append(name)
    print "Score %s: %.3f%%, %.3f%%" % (name, cv_res.mean()*100, cv_res.std()*100)


# In[ ]:




# In[ ]:




# ## Feature engineering

# ### TODO
# - Travail d'une force : vitesse * direction  
# https://fr.wikipedia.org/wiki/Travail_d%27une_force
# 

# # Draft

# In[29]:

groupby_cities = file_.groupby('insee')


# Moving Average

# In[30]:

pd.DataFrame(groupby_cities.tH2_obs.rolling(window=2).mean())


# Interpolate NA val

# In[ ]:

groupby_city.capeinsSOL0.apply(pd.Series.interpolate)


# # Documentation
# 
# - Handling missing val for RF  
# https://stats.stackexchange.com/questions/98953/why-doesnt-random-forest-handle-missing-values-in-predictors/186264#186264  https://github.com/scikit-learn/scikit-learn/issues/5870  

# In[ ]:



