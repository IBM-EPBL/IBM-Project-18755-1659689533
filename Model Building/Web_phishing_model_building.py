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


# In[18]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[19]:


y_pred1=lr.predict(x_test)
from sklearn.metrics import accuracy_score
log_reg=accuracy_score(y_test,y_pred1)
log_reg


# In[20]:


import pickle
pickle.dump(lr,open('Phishing_Website.pkl','wb'))


# In[ ]:




