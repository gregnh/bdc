
# coding: utf-8

# # TODO
# - feature engineering sur la datetime (add year, season, weeknumber)
# - LSTM w/ Keras
# - ARIMA & SARIMA Model
# - use KNN to interpolate

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd, numpy as np
import sys, os

import sklearn.preprocessing as pp
import sklearn.decomposition as decomposition
import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error

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
    return pd.concat([df, ohc], axis=1)


# In[20]:

def processing(filename = 'train_1', drop_method = 'any'):
    data = pd.read_csv(raw_path + filename + '.csv', header=0, delimiter=';',decimal=',',
                        parse_dates=['date'], index_col='date')
    
    data = data.rename(columns={'mois':'month'})
    raw_colnames = list(data.columns.values)

    #Transform to dummy
    dummies = ['ddH10_rose4', 'insee', 'month']
    df_dummy = pd.get_dummies(data, columns=dummies, prefix=dummies)
    data.drop(['ddH10_rose4'], inplace=True, axis=1)

    # Add temporal features
    # data['week'] = data.index.week # WEEK AS CATEGORY ?
    data = season(data) # add season 
    data.drop(['month'], inplace=True, axis=1)

    # Add lag operator (shift ce fait par ville)
    groupby_cities = data.groupby('insee')
    shift = groupby_cities.tH2_obs.shift(1)
    data['tH2_obs_lag1'] = shift
    
    data.dropna(how=drop_method, axis=0, inplace=True)
    data.drop(['insee'], inplace=True, axis=1)
    return data, [x for x in raw_colnames if x not in dummies]
file_, raw_col = processing()
#assert nb de col is good


# In[5]:

file_.describe(include='all')


# In[8]:

file_.head(10)


# ## Factor Analysis

# In[7]:

# file_['2014-01'][]
# calculer la variation moyenne de temperature en de décembre à janvier pour estimer 2014-01-01


# Normalize float data

# In[ ]:

pp.Normalizer().fit_transform(file_[raw_col]) #normalize only original float columns


# In[98]:

# assert no nan
fa = decomposition.FactorAnalysis().fit(file_.drop('tH2_obs'))


# In[99]:

fa


# ## ML

# In[ ]:

seed = 1
results_dict = {{}}
predictions_dict = {{}}
results = []
predictions = []


# In[ ]:

models = []
models.append(('CART', tree.DecisionTreeRegressor()))
models.append(('RF', ensemble.RandomForestRegressor(n_jobs=3)))
models.append(('GB', ensemble.GradientBoostingRegressor()))
# models.append(('NB'), ) # Naive Bayes for modeling uncertainty
# LSTM
# xgboost

# poor results
# models.append(('KNN', neighbors.KNeighborsRegressor(n_jobs=3)))
# models.append(('SVM', svm.SVR()))

# for name, model in models:
#     results_dict[name] = {}
    


# Apply forward walk

# In[ ]:

from sklearn.metrics import mean_squared_error

first_train = True
for i in range(1,36):
    print(i)
    train_file = 'train_' + str(i)
    test_file = 'train_' + str(i+1)
    
    x_test, tmp = processing(filename=test_file)
    y_test = pd.DataFrame(x_test.tH2_obs)
    x_test = x_test.drop('tH2_obs', axis=1)
    
    train, tmp = processing(train_file)
    if first_train is True:
        x_train = train.drop('tH2_obs', axis=1)
        y_train = train.pop('tH2_obs')
        first_train = False
    else:
        x_train = pd.concat([x_train, train.drop('tH2_obs', axis=1)])
        y_train = pd.concat([y_train, train.pop('tH2_obs')])
        
    for name, model in models: 
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        result = np.sqrt(mean_squared_error(y_test, pred))
        
        # revoir structure
        results_dict[name][i] = result
        predictions_dict[name][i] = pred
        print(name, ' : ', result)


# In[ ]:

# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black",
#             c="darkorange", label="data")
# plt.plot(x_test, pre, color="cornflowerblue",
#          label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()


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

pd.DataFrame(groupby_cities.tH2_obs.rolling(window=2).mean()) # ?


# Interpolate NA val

# In[ ]:

groupby_city.capeinsSOL0.apply(pd.Series.interpolate)


# # Documentation
# 
# - Handling missing val for RF  
# https://stats.stackexchange.com/questions/98953/why-doesnt-random-forest-handle-missing-values-in-predictors/186264#186264  https://github.com/scikit-learn/scikit-learn/issues/5870  

# In[ ]:



