#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
import datetime
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("supermarket_sales - Sheet1.csv")
data


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.value_counts()


# In[8]:


data.shape


# In[9]:


data.dtypes


# In[10]:


data.columns


# # Checking Null Value

# In[11]:


data.isnull().sum()


# In[12]:


data.isnull().any()


# # Exploratory Data Analysis

# HISTOGRAM
# 
# 

# In[13]:


data.hist(figsize=(20,14))
plt.show()


# 
# BOXPLOT

# In[16]:


plt.figure(figsize=(14,10))
sns.set_style(style='whitegrid')
plt.subplot(2,3,1)
sns.boxplot(x='Unit price',data=data)
plt.subplot(2,3,2)
sns.boxplot(x='Quantity',data=data)
plt.subplot(2,3,3)
sns.boxplot(x='Total',data=data)
plt.subplot(2,3,4)
sns.boxplot(x='cogs',data=data)
plt.subplot(2,3,5)
sns.boxplot(x='Rating',data=data)
plt.subplot(2,3,6)
sns.boxplot(x='gross income',data=data)


# PAIRPLOT

# In[17]:


sns.pairplot(data=data)


# REGPLOT

# In[18]:


sns.regplot(x='Rating', y= 'gross income', data=data)


# SCATTER PLOT
# 
# 

# In[19]:


sns.scatterplot(x='Rating', y= 'cogs', data=data)


# JOINTPLOT
# 
# 

# In[20]:


sns.jointplot(x='Rating', y= 'Total', data=data)


# CATPLOT

# In[21]:


sns.catplot(x='Rating', y= 'cogs', data=data)


# LMPLOT

# In[22]:


sns.lmplot(x='Rating', y= 'cogs', data=data)


# In[23]:


data.columns


# KDE PLOT (DENSITY PLOT)

# In[24]:


plt.style.use("default")

sns.kdeplot(x='Rating', y= 'Unit price', data=data)


# LINEPLOT

# In[25]:


sns.lineplot(x='Rating', y= 'Unit price', data=data)


# BARPLOT

# In[26]:


plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Unit price", data=data[170:180])
plt.title("Rating vs Unit Price",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Unit Price")
plt.show()


# In[27]:


data.columns


# In[28]:


plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Gender", data=data[170:180])
plt.title("Rating vs Gender",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Gender")
plt.show()


# In[29]:


plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Quantity", data=data[170:180])
plt.title("Rating vs Quantity",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Quantity")
plt.show()


# In[30]:


#lets find the categorialfeatures
list_1=list(data.columns)
list_cate=[]
for i in list_1:
    if data[i].dtype=='object':
        list_cate.append(i)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in list_cate:
    data[i]=le.fit_transform(data[i])
data


# In[31]:


y=data['Gender']
x=data.drop('Gender',axis=1)


# # TRAINING AND TESTING DATA
# 

# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# # MODELS

# SVC

# In[35]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)


# In[36]:


y_pred=svc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",svc.score(x_train,y_train)*100)


# Naive Bayes

# In[37]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)


# In[38]:


y_pred=gnb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",gnb.score(x_train,y_train)*100)


# DECISION TREE CLASSIFIER

# In[39]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)


# In[40]:


y_pred=dtree.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",dtree.score(x_train,y_train)*100)


# Random Forest Classifier

# In[41]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)


# In[42]:


y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(x_train,y_train)*100)


# AdaBoostClassifier
# 
# 
# 

# In[43]:


from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(base_estimator = None)
adb.fit(x_train,y_train)


# In[44]:


y_pred=adb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",adb.score(x_train,y_train)*100)


# Gradient Boosting Classifier

# In[45]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)


# In[46]:


y_pred=gbc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",gbc.score(x_train,y_train)*100)


# In[47]:


data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
data


# ExtraTreesClassifier
# 
# 

# In[49]:


from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
etc.fit(x_train,y_train)


# In[50]:


y_pred=etc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",etc.score(x_train,y_train)*100)


# Bagging Classifier

# In[51]:


from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
model.score(x_test,y_test)


# In[52]:


data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
data


# In[ ]:




