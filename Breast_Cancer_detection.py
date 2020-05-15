#!/usr/bin/env python
# coding: utf-8

# # Importing The Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the Dataset

# In[4]:


from sklearn.datasets import load_breast_cancer


# In[6]:


cancer = load_breast_cancer()


# In[8]:


cancer


# In[10]:


cancer.keys()


# In[45]:


print(cancer['feature_names'])


# In[19]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# In[35]:


df_cancer.loc[1:20]


# # Visulaizing the data

# In[46]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])


# In[49]:


sns.countplot(df_cancer['target'])


# In[52]:


sns.scatterplot(x = 'mean perimeter', y = 'mean texture', hue = 'target', data = df_cancer)


# In[59]:


plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(), annot = True)


# In[64]:


x = df_cancer.drop(['target'], axis = 1)


# In[66]:


y = df_cancer['target']


# In[69]:


from sklearn.model_selection import train_test_split


# In[71]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 5)


# In[72]:


x_train


# In[73]:


x_test


# In[98]:


min_train = x_train.min()


# In[99]:


range_train = (x_train - min_train).max()


# In[100]:


x_train_sc = (x_train - min_train)/range_train


# In[104]:


sns.scatterplot(x = x_train['mean perimeter'], y = x_train['mean texture'], hue = y_train)


# In[105]:


sns.scatterplot(x = x_train_sc['mean perimeter'], y = x_train_sc['mean texture'], hue = y_train)


# In[106]:


min_test = x_test.min()
range_test = (x_test - min_test).max()
x_test_sc = (x_test - min_test)/range_test


# In[75]:


from sklearn.svm import SVC


# In[86]:


get_ipython().run_line_magic('pinfo', 'SVC')


# In[126]:


classifier = SVC()
classifier.fit(x_train_sc, y_train)


# # Model Evaluation

# In[113]:


from sklearn.metrics import confusion_matrix, classification_report


# In[108]:


y_pred = classifier.predict(x_test_sc)


# In[109]:


y_pred


# In[111]:


cm = confusion_matrix(y_test, y_pred)


# In[112]:


sns.heatmap(cm, annot = True)


# In[115]:


print(classification_report(y_test, y_pred))


# In[142]:


param_grid = {'C' : [0.1, 1, 10, 100], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel' : ['rbf']}


# In[143]:


from sklearn.model_selection import GridSearchCV


# In[144]:


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)


# In[145]:


grid.fit(x_train_sc, y_train)


# In[146]:


grid.best_params_


# In[133]:


grid_pred = grid.predict(x_test_sc)


# In[134]:


cm = confusion_matrix(y_test, grid_pred)


# In[147]:


sns.heatmap(cm, annot = True)


# In[152]:


print(classification_report(y_test, grid_pred))

