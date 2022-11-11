#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score


# In[12]:


spreadsheet = pd.read_csv('C:/Users/ritha/Downloads/dataset_website.csv')


# In[13]:


spreadsheet.head()


# In[14]:


spreadsheet.info()
spreadsheet.isnull().any()


# In[15]:


x=spreadsheet.iloc[:,1:31].values
y=spreadsheet.iloc[:,-1].values
print(x,y)


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:




