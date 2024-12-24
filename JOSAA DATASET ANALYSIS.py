#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

newdf = pd.read_csv('2016.csv')

print(newdf.head())


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

f3 = pd.read_csv('2016.csv')
f4 = pd.read_csv('2017.csv')
f5 = pd.read_csv('2018.csv')
train = pd.concat([f3,f4,f5])
test = pd.read_csv('2019.csv')
total = pd.concat([train,test])
train.head(2)


# In[10]:


train.isnull().sum()


# In[11]:


train.info()


# In[12]:


train = train.dropna()
test = test.dropna()
non_integers = list(set(train["Opening Rank"][~train["Opening Rank"].apply(lambda x: str(x).isdigit())]))
print(non_integers)


# In[13]:


import re
def to_numeric(x):
    x = str(x)
    x = x.replace(",", "")
    x = re.sub("[a-zA-Z]", "", x)
   
    return float(x) if '.' in x else int(x)

# Convert train data columns to numeric and integer types
train["Opening Rank"] = train["Opening Rank"].apply(to_numeric)
train["Closing Rank"] = train["Closing Rank"].apply(to_numeric)
train["Opening Rank"] = train["Opening Rank"].astype(int)
train["Closing Rank"] = train["Closing Rank"].astype(int)

# Convert test data columns to numeric and integer types
test["Opening Rank"] = test["Opening Rank"].apply(to_numeric)
test["Closing Rank"] = test["Closing Rank"].apply(to_numeric)
test["Opening Rank"] = test["Opening Rank"].astype(int)
test["Closing Rank"] = test["Closing Rank"].astype(int)

# Check for non-integer values in the train data
non_integers = list(set(train["Opening Rank"][~train["Opening Rank"].apply(lambda x: str(x).isdigit())]))
print(len(non_integers))


# In[14]:


train.info()


# In[15]:


train.isnull().sum()


# In[23]:


x_train = train.drop(["Academic Program Name", "Institute"], axis=1)
x_test = test.drop(["Academic Program Name", "Institute"], axis=1)
total = pd.concat([train,test])
total.corr()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
#sns.heatmap(data=df.corr(),annot=True, cmap="coolwarm")
corr_matrix = total.corr()

# Create heatmap with larger size
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix of Dataset")


# In[24]:


import matplotlib.pyplot as plt
total.hist(figsize=(5,8))
plt.show()


# In[25]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler
columns_to_scale = ['Opening Rank', 'Closing Rank', 'Year','Round']
scaler = StandardScaler()
scaler.fit(train[columns_to_scale])
train[columns_to_scale] = scaler.transform(train[columns_to_scale])
test[columns_to_scale] = scaler.transform(test[columns_to_scale])

train.head(2)


# In[26]:


total = pd.concat([train,test])
corr_matrix = total.corr()

# Create heatmap with larger size
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix of Dataset")


# In[ ]:




