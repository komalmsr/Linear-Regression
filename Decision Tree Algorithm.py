#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Algorithm
# 
# 
# For the given ‘Iris’ dataset, the task is to create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# ### Importing Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn.datasets


# ### Loading the Dataset

# In[7]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[9]:


# Forming the iris dataframe
df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head(5))


# In[10]:


df.shape


# There are 150 rows and 4 columns in this dataset

# ### Splitting the Data

# In[11]:


x = iris.data
y = iris.target


# In[12]:


y


# ### Training and Testing Data

# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2)


# In[14]:


x_train.shape


# In[15]:


x_test.shape


# ### Basic Decision Tree Model

# In[16]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree


# In[17]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


# ### Checking for Accuracy

# In[18]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# We got a classification rate of 90%, considered as good accuracy. We can improve this accuracy by tuning the parameters in the Decision Tree Algorithm.

# ### Visualizing Decision Trees

# In[19]:


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)


# In[ ]:




