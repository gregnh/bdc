
# coding: utf-8

# # TODO
# - interpolate missing values

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
import sys
sys.path.append('../')
import lib.tools as tools

pd.options.display.max_columns = 50
raw_path = '../data/raw/'
train_path = '../data/train/'


# In[2]:

results = [] #dict()
preds = [] #dict()
# for i in range(1,37):
#     results[i] = []
#     preds[i] = []


# In[3]:

models = dict()
for i in range(1,37):
    models[i] =  xgb.XGBRegressor(max_depth=3, learning_rate=0.1, nthread=3, )
model = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, nthread=3, )


# In[ ]:

def ts_split(index_series, n_splits=4):
    len_split = len(index_series) // n_splits
    return [len_split*split_number for split_number in range(n_splits)] 


# In[19]:

x_total = None
y_total = None
# date_total = None
# Epoch
for filenumber in range(1,36+1):
    filename = 'train_' + str(filenumber) + '.csv'
    print(filename)
    data = pd.read_csv(train_path + filename, header=0, delimiter=';', parse_dates=['date'])
    if x_total is not None:
        data.index = data.index + x_total.index[-1]
        print(data.index[-1])
    y = pd.DataFrame(data.tH2_obs)
    x = data.drop(['tH2_obs', 'date'], axis=1)

    x_total = pd.concat([x_total, x], axis=0, ignore_index=True)
    y_total = pd.concat([y_total, y], axis=0, ignore_index=True)
    
    # Batch
#     tssplit = TimeSeriesSplit(n_splits=5)
    for split_index in ts_split(data.index, n:
#         print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x_total.iloc[train_index,:], x_total.iloc[test_index,:]
        y_train, y_test = y_total.iloc[train_index,:], y_total.iloc[test_index,:]
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        result = np.sqrt(mean_squared_error(y_test, pred))

        preds.append(pred)
        results.append(result)
        print('RMSE:', result)
    


# In[15]:

listtrain_index[:2]


# In[ ]:

plt.figure(dpi=100)
xgb.plot_importance(model, )
plt.show()


# In[ ]:

test_data = pd.read_csv('../data/test/test.csv', header=0, parse_dates=['date'], delimiter=';', index_col='date', )
test_ = tools.processing(test_data)


# In[ ]:

model.predict(test_)


# In[ ]:

# # Epoch
# # x_total = None
# # y_total = None
# # date_total = None
# for filenumber in range(1,36+1):
#     filename = 'train_' + str(filenumber) + '.csv'
#     print(filename)
#     # Batch
#     data = pd.read_csv(train_path + filename, header=0, delimiter=';', parse_dates=['date'])
#     y = pd.DataFrame(data.tH2_obs)
#     x = data.drop(['tH2_obs', 'date'], axis=1)

# #     x_total = pd.concat([x_total, x], axis=0, ignore_index=True)
# #     y_total = pd.concat([y_total, y], axis=0, ignore_index=True)
    
#     tssplit = TimeSeriesSplit(n_splits=4)
#     for train_index, test_index in tssplit.split(x):
# #         print("TRAIN:", train_index, "TEST:", test_index)
#         x_train, x_test = x.iloc[train_index,:], x.iloc[test_index,:]
#         y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
        
#         models[filenumber].fit(x_train, y_train)
#         pred = models[filenumber].predict(x_test)
#         result = np.sqrt(mean_squared_error(y_test, pred))

#         preds[i].append(pred)
#         results[i].append(result)
#         print('RMSE:', result)
    


# In[57]:

len(y_total.index)


# In[ ]:



