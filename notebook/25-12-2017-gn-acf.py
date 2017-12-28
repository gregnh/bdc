
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sys

sys.path.append('../')

pd.options.display.max_columns = 50


# In[42]:

data_path = '../data/'
train = pd.read_csv(data_path + "raw/train_1.csv", header=0, delimiter=";",decimal=',',
                    parse_dates=['date'], index_col='date')
# train.dropna(how='any', inplace=True)


# In[3]:

used_var = ['ffH10', 'flir1SOL0', 'fllat1SOL0', 'flsen1SOL0',
       'flvis1SOL0', 'hcoulimSOL0', 'huH2', 'iwcSOL0', 'nbSOL0_HMoy',
       'nH20', 'ntSOL0_HMoy', 'pMER0', 'rr1SOL0', 'tH2',
            'tH2_VGrad_2.100', 'tH2_XGrad', 'tH2_YGrad', 'tpwHPA850', 'ux1H10',
            'vapcSOL0', 'vx1H10']


# In[20]:

def autocorrelation(x_array, t=1):
    return x_array.autocorr(t)


# In[43]:

lag = 90
acf = [autocorrelation(train.flsen1SOL0, i) for i in range(1,lag)]

plt.figure(dpi=100)
plt.vlines(range(1,lag), 0, acf)
plt.plot([0.2]*lag, '--')
plt.plot([-0.2]*lag, '--')
plt.show()


# # Conclusion
# 
# - Mostly unsignificant but low (0,3-0.4)autocorelation at lag 1 and 7 : ffH10, 
# 
# - Mostly unsignificant but low autocorelation at lag 7 : flir1SOL0
# 
# - Mostly unsignificant but medium autocorelation at lag t*7 : fllat1SOL0
# 
# .
# .
# .

# In[ ]:



