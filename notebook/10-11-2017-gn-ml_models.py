
# coding: utf-8

# # TODO
# - feature engineering sur la datetime (add year, season, weeknumber)
# - LSTM w/ Keras
# - ARIMA & SARIMA Model
# - use KNN to interpolate

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import sys, os

import sklearn.preprocessing as pp
import sklearn.decomposition as decomposition
# import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error

# import sklearn.svm as svm
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
import xgboost as xgb

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

used_var = ['ffH10', 'flir1SOL0', 'fllat1SOL0', 'flsen1SOL0',
       'flvis1SOL0', 'hcoulimSOL0', 'huH2', 'iwcSOL0', 'nbSOL0_HMoy',
       'nH20', 'ntSOL0_HMoy', 'pMER0', 'rr1SOL0', 'tH2',
            'tH2_VGrad_2.100', 'tH2_XGrad', 'tH2_YGrad', 'tpwHPA850', 'ux1H10',
            'vapcSOL0', 'vx1H10']


# In[4]:

def processing(filename = 'train_1', drop_method = 'any'):
    data = pd.read_csv(raw_path + filename + '.csv', header=0, delimiter=';',decimal=',',
                        parse_dates=['date'], index_col='date')
    
    data = data.rename(columns={'mois':'month'})
#     raw_colnames = list(data.columns.values)

    # Add lag operator (shift ce fait par ville)
    groupby_cities = data.groupby('insee')
    shift = groupby_cities.tH2.shift(1)
    data['tH2_lag1'] = shift
    
    # Add temporal features
    # data['week'] = data.index.week # WEEK AS CATEGORY ?
    data = season(data) # add season 
    data.drop(['month'], inplace=True, axis=1)
    
    # Variables to be dropped
    data.drop(['capeinsSOL0', 'ciwcH20', 'clwcH20'], inplace=True, axis=1)
    
    #Transform to dummy
    dummies = ['ddH10_rose4', 'insee', 'month']
    df_dummy = pd.get_dummies(data, columns=dummies, prefix=dummies)
    data.drop(['ddH10_rose4'], inplace=True, axis=1)
    
    # Drop NAN
    data.dropna(how=drop_method, axis=0, inplace=True)
    data.drop(['insee'], inplace=True, axis=1)
    return data
file_ = processing()
#assert nb de col is good


# In[5]:

file_.info()


# In[5]:

file_.describe(include='all')


# In[6]:

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

# In[7]:

seed = 1
results_dict = dict()
predictions_dict = dict()


# In[9]:

models = []
# models.append(('CART', tree.DecisionTreeRegressor()))
models.append(('RF', ensemble.RandomForestRegressor(n_estimators=40, n_jobs=3,bootstrap=False)))
models.append(('GB', ensemble.GradientBoostingRegressor(n_estimators=40, )))
# models.append(('NB'), ) # Naive Bayes for modeling uncertainty
# LSTM
# models.append(('XGB', xgb.XGBRegressor()))

# poor results
# models.append(('KNN', neighbors.KNeighborsRegressor(n_jobs=3)))
# models.append(('SVM', svm.SVR()))

# for name, model in models:
#     results_dict[name] = {}
    


# Apply forward walk

# In[10]:

first_train = True
for i in range(1,36): # epochs
    print(i)
    train_file = 'train_' + str(i)
    test_file = 'train_' + str(i+1)
    
    x_test= processing(filename=test_file)
    y_test = pd.DataFrame(x_test['data'].tH2_obs)
    x_test['data'] = x_test['data'].drop('tH2_obs', axis=1)
    
    train = processing(train_file)
    if first_train is True:
        x_train = train['data'].drop('tH2_obs', axis=1)
        y_train = train['data'].pop('tH2_obs')
        first_train = False
    else:
        x_train = pd.concat([x_train, train['data'].drop('tH2_obs', axis=1)])
        y_train = pd.concat([y_train, train['data'].pop('tH2_obs')])
        
    for name, model in models: 
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        result = np.sqrt(mean_squared_error(y_test, pred))
        
        # revoir structure
        if name in results_dict:
            results_dict[name].append(result)
            predictions_dict[name].append(pred)
        else:
            results_dict[name] = [result]
            predictions_dict[name] = [pred]
        print(name, ' : ', result)


# In[36]:

plt.figure(dpi=100)
for key, val in results_dict.iteritems():
    plt.plot(range(len(val)), val, '-', label=key)
plt.ylabel("RMSE")
plt.xlabel("Nb of dataset used for training")
plt.legend()
plt.show()


# I dont keep CART, SVM, KNN  
# KNN and SVM : computational time too long  
# CART : results not satisfying

# # Feature importance

# In[38]:

plt.figure(dpi=90, figsize=(10, 10))
imp = models[1][1].feature_importances_

imp, names = zip(*sorted(zip(imp, x_train.columns.values)))

plt.barh(range(len(names)),imp,align='center')
plt.yticks(range(len(names)),names)
plt.xlabel('Importance of features')
plt.ylabel('Features')
plt.show()

plt.show()


# In[29]:

for imp_, name in zip(imp, names):
    print(name, imp_)


# low importance : 
# - capeinsSOL0
# - ciwcH20
# - clwcH20
# - ffH10
# - flir1SOL0
# - hcoulimSOL0
# - iwcSOL0
# - rr1SOL0
# - rrH20
# - tH2_XGrad
# - tH2_VGrad_2.100
# - nH20
# - fllat1SOL0
# - ux1H10

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



