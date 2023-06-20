#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch

import tools


# In[ ]:


n = 100000
dy = 1
num_cat_features = 10
num_cont_features = 30
feature_cols = [f'x{n}' for n in range(num_cat_features + num_cont_features)]
cat_features = feature_cols[:num_cat_features]
float_features = feature_cols[num_cat_features:]
targets = [f'y{n}' for n in range(dy)]


# In[ ]:


data = pd.read_csv('data/large.csv')
xdf = data.loc[:, feature_cols]
x = xdf.values
ydf = data.loc[:, targets]
y = ydf.values
store = pickle.load(open('data/store.exp2', 'rb'))


# In[ ]:


expected_cat = store['expected_cat']
expected_cont0  = store['expected_cont0']
expected_cont1  = store['expected_cont1']
expected_cont = store['expected_cont']
expected_features = store['expected_features']


# ### Uncover relation between features and data

# In[ ]:


_chooser = data.iloc[:, expected_cat[1]] == data.iloc[:, expected_cat[0]]
idx0 = _chooser == 0
idx1 = _chooser == 1
y_ = np.zeros(shape=(len(data), dy))
y_[idx0, :] = (
    store['t0'] @ np.expand_dims(
        np.sin(2 * np.pi * data.loc[idx0].iloc[:, expected_cont0]),
axis=2))[:, :, 0]
y_[idx1, :] = (
    store['t1'] @ np.expand_dims(
        np.cos(2 * np.pi * data.loc[idx1].iloc[:, expected_cont1]),
axis=2))[:, :, 0]


# In[ ]:


assert np.allclose(np.squeeze(y_), data['y0'].values, atol=1e-6, rtol=1e-4)


# ### Selection with Boruta

# In[ ]:


from arfs.feature_selection import allrelevant
from arfs.feature_selection.allrelevant import Leshy
from xgboost import XGBRegressor


# In[ ]:


n_estimators = 'auto'
importance = "native"
max_iter = 100
random_state = None
verbose = 0
keep_weak = False


# In[ ]:


xdf = pd.DataFrame(x, columns = [f'f{i}' for i in range(num_cat_features + num_cont_features)])
yser = pd.Series(y[:, 0], name='y')


# In[ ]:


regressor = XGBRegressor(random_state=42)


# In[ ]:


leshy = Leshy(
    regressor,
    n_estimators=n_estimators,
    importance=importance,
    max_iter=max_iter,
    random_state=random_state,
    verbose=verbose,
    keep_weak=keep_weak,
)


# In[ ]:


leshy.fit(xdf, yser)
leshy_selection = [int(col.replace('f', '')) for col in leshy.selected_features_]


# In[ ]:


print(f'Expected features: {sorted(expected_features)}')
print(f'Boruta selection: {sorted(leshy_selection)}')

