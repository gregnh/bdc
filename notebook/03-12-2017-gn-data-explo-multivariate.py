
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib notebook')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sys

sys.path.append('../')
data_path = '../data/raw/'


# In[3]:

data = pd.read_csv(data_path + 'train_1.csv', header=0, delimiter=";",decimal=',',
                    parse_dates=['date'], index_col='date')
data = data.drop(['insee', 'ech', 'mois'], axis=1)


# In[4]:

data.columns.values


# # Correlation between variables 

# In[5]:

corr = data.corr()

#Mask upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sb.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sb.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:



