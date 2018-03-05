
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

pd.options.display.max_columns = 35
raw_path = '../data/raw/'


# In[38]:

train = pd.read_csv(raw_path + "train_1.csv", header=0, delimiter=";",decimal=',',
                    parse_dates=['date'], index_col='date')
train.head()


# In[39]:

train.drop(axis=1, columns=['capeinsSOL0', 'ciwcH20', 'clwcH20', 'ddH10_rose4', 'rr1SOL0', 'rrH20', 'ffH10', 
                            'flir1SOL0', 'hcoulimSOL0','huH2', 'nH20', 'iwcSOL0', 'tH2_XGrad'], 
           inplace=True)

print(train.columns)


# ## Check where the missing vals are

# In[40]:

# Total missing val per var
pd.DataFrame(train.isnull().sum()).T


# In[18]:

col = np.where(train.isnull().sum() == 140)[0]
print(np.where(np.asanyarray(pd.isnull(train.iloc[:,col])
                            )
              )[0]
     )


# In[32]:

pd.DataFrame(train.iloc[2653-7*2, :]).T


# ## Lets see if one can interpolate the missing values

# In[44]:

for i in sorted(range(1,12), reverse=True):
    print(train.index[2653-7*i].date(), 
          '- tH2 val : {} for station {}'.format(train.pMER0[2653-7*i], train.insee[2653-7*i]))
print('interpolate : {}'.format(train.pMER0.interpolate()[2653]))


# In[70]:

for i in range(1,20):
    print(train.index[2653+7*i].date(), train.pMER0[2653+7*i])


# In[37]:

gb_cities = train.groupby('insee')
print(gb_cities.pMER0.resample('W').median()[6088001]['2015-01'])
# print(gb_cities.index.week[2653-7*3])


# Seems to work when replacing with median with respect to insee

# In[68]:

for i in sorted(range(1,5), reverse=True):
    print(train.index[4627-7*i].date(), 
          ' - pMER0 val : {} for station {}'.format(train.pMER0[4627-7*i], train.insee[4627-7*i]))
print('interpolate : {}'.format(train.pMER0.interpolate(method='krogh')[4627]))
for i in range(1,5):
    print(train.pMER0[4627+7*i])


# In[114]:

gb_cities.pMER0.apply(lambda group: group.interpolate(method='time'))[6088001]


# In[118]:

# Interpolate on every variable for a city
sum(pd.isnull(train[train['insee'] == 6088001].apply(lambda group: group.interpolate(method='time'))).sum())


# https://machinelearningmastery.com/time-series-data-visualization-with-python/

# In[ ]:



