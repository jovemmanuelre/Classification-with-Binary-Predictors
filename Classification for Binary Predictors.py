#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


# In[4]:


raw_data = pd.read_csv('2.02. Binary predictors.csv')
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
data


# In[5]:


y = data['Admitted']
x1 = data[['SAT','Gender']]


# ## Regression

# In[6]:


x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()


# In[7]:


np.exp(1.9449) 
# Given the same SAT score, a female is 7 times more likely to get admitted than a male.


# ## Training the Model

# In[8]:


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
# Used this formatting to show a more logical result.
results_log.predict()
# These are the Predicted Values of my model


# In[9]:


np.array(data['Admitted'])
# While these are the actual values of the data.


# In[10]:


results_log.pred_table()
# I then compare the Predicted Values of the model I trained and the Actual Values using the Confusion Matrix.


# In[11]:


cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df


# In[12]:


cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
accuracy_train
# The accuracy of my model using the training data is 94.6%


# ## Testing the Model and assessing its accuracy

# In[13]:


test = pd.read_csv('2.03. Test dataset.csv')
test
# In this exercise the data was split into 90-10 beforehand.


# In[14]:


test['Admitted'] = test['Admitted'].map({'Yes': 1, 'No': 0})
test['Gender'] = test['Gender'].map({'Female': 1, 'Male': 0})
test


# In[15]:


x
# This is the Regression variable I defined earlier that contains independent variables and their constant.


# In[16]:


test_actual = test['Admitted']
test_data = test.drop(['Admitted'],axis=1)
test_data = sm.add_constant(test_data)
test_data
# Decided to drop 'Admitted' because Order is important. The coefficients of the Regression expect it.


# In[17]:


def confusion_matrix(data,actual_values,model):
    
        pred_values = model.predict(data)
        bins=np.array([0,0.5,1])
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()
        return cm, accuracy


# In[18]:


cm = confusion_matrix(test_data,test_actual,results_log)
cm
# shows the Confusion Matrix and the Accuracy
# The accuracy is 89.47%


# In[19]:


cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df
# Here I made a more presentable Confusion Matrix of my data.


# In[20]:


print ('Missclassification rate: '+str((1+1)/19))
# The opposite of accuracy is the misclassification rate.
# It is 10.53&

