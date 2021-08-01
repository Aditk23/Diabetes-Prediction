#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# In[2]:


#Importing dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names=['Pregnancies','Glucose','BP','SkinThickness','Insulin','BMI','DiabetesPedigree','Age','Outcome']
data = pd.read_csv(url,names=names)


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


#Checking for null values, none found
data.isnull().sum()


# In[6]:


data.info()


# In[52]:


sns.countplot(data.Outcome)


# In[7]:


sns.distplot(data['Pregnancies'])


# In[8]:


sns.boxplot(data['Pregnancies'])


# In[9]:


data.loc[data.Pregnancies>13]


# In[10]:


data.drop(data[data['Pregnancies'] > 13].index, inplace = True)
data.shape


# In[11]:


# On some rows glucose value is 0
sns.boxplot(data.Glucose)


# In[12]:


data.loc[data.Glucose==0]


# In[13]:


# As there are only 4 rows with ) glucose value deleting these rows
data.drop(data[data['Glucose']==0].index, inplace = True)
data.shape


# In[14]:


sns.distplot(data.Glucose)


# In[15]:


# For many rows BP is 0. But 0 BP does not make any sense so it might be the case that for these people BP value was
# not available.
sns.distplot(data.BP)


# In[16]:


data.loc[data.BP==0].shape


# In[17]:


# As this is small dataset replacing 0 BP values with median
data.BP = data.BP.replace(0,data.BP.median())


# In[18]:


# Outliers are still present for BP
sns.boxplot(data.BP)


# In[19]:


data.loc[(data.BP<=40)|(data.BP>=113)]


# In[20]:


data.drop(data[data.BP<=40].index, inplace = True)
data.drop(data[data.BP>=113].index, inplace = True)

data.shape


# In[21]:


sns.distplot(data.SkinThickness)


# In[22]:


data.loc[data['SkinThickness']==0].shape


# In[23]:


# Out of 752 rows 225 have skinthickness as 0 so dropping this column
data.drop('SkinThickness',axis=1,inplace=True)


# In[24]:


data.shape


# In[25]:


sns.distplot(data.Insulin)


# In[26]:


data_ins = data.loc[data.Insulin!=0]
data_ins.shape


# In[27]:


data_ins.loc[data_ins.Outcome==1].Insulin.median()


# In[28]:


data_ins.loc[data_ins.Outcome==0].Insulin.median()


# In[29]:


data.loc[(data['Outcome'] == 0 ) & (data['Insulin']==0), 'Insulin'] = 105.0
data.loc[(data['Outcome'] == 1 ) & (data['Insulin']==0), 'Insulin'] = 175.0


# In[30]:


sns.distplot(data.BMI)


# In[31]:


data.loc[data.BMI==0]


# In[32]:


data.BMI.median()


# In[33]:


data.BMI = data.BMI.replace(0,data.BMI.median())


# In[34]:


sns.boxplot(data.BMI)


# In[35]:


data.loc[data.BMI>50]


# In[36]:


data.drop(data[data.BMI>50].index, inplace = True)


# In[37]:


data.shape


# In[38]:


sns.distplot(data['DiabetesPedigree'])


# In[39]:


data.head()


# In[44]:


data.DiabetesPedigree = np.sqrt(data.DiabetesPedigree)


# In[45]:


sns.distplot(data.DiabetesPedigree)


# In[46]:


sns.boxplot(data.DiabetesPedigree)


# In[47]:


data.drop(data[data.DiabetesPedigree>1.2].index, inplace = True)
data.shape


# In[50]:


sns.boxplot(data.Age)


# In[51]:


data.drop(data[data.Age>65].index, inplace = True)
data.shape


# In[53]:


sns.countplot(data.Outcome)


# In[56]:


sns.scatterplot(data=data, x="Glucose", y="Age", hue="Outcome")
# With increase in age glucose increases and chances of diabetes increases


# In[58]:


sns.heatmap(data.corr(), fmt='.2g', annot=True, cmap = 'YlGnBu')

#No significant correlation


# In[59]:


data.head()


# In[60]:


feature_space = data.iloc[:, data.columns != 'Outcome']
feature_class = data.iloc[:, data.columns == 'Outcome']


# In[62]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 

training_set, test_set, class_set, test_class_set = train_test_split(feature_space,
                                                                    feature_class,
                                                                    test_size = 0.20, 
                                                                    random_state = 42)


# In[63]:


fit_rf = RandomForestClassifier(random_state=42)


# In[65]:


import time

np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [2, 3, 4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(fit_rf, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 3)
cv_rf.fit(training_set, class_set)
print('Best Parameters using grid search: \n', cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))


# In[66]:


# Set best parameters given by grid search 
fit_rf.set_params(bootstrap=True,
                  criterion = 'entropy',
                  max_features = None, 
                  max_depth = 4)


# In[67]:


fit_rf.set_params(warm_start=True, 
                  oob_score=True)

min_estimators = 15
max_estimators = 1000

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    fit_rf.set_params(n_estimators=i)
    fit_rf.fit(training_set, class_set)

    oob_error = 1 - fit_rf.oob_score_
    error_rate[i] = oob_error


# In[68]:


# Convert dictionary to a pandas series for easy plotting 
oob_series = pd.Series(error_rate)


# In[69]:


fig, ax = plt.subplots(figsize=(10, 10))

ax.set_facecolor('#fafafa')

oob_series.plot(kind='line',color = 'red')
plt.axhline(0.055, color='#875FDB',linestyle='--')
plt.axhline(0.05, color='#875FDB',linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')


# In[70]:


# Refine the tree via OOB Output
fit_rf.set_params(n_estimators=200,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)


# In[71]:


fit_rf.fit(training_set, class_set)


# In[ ]:


predictions_rf = fit_rf.predict(test_set)


# In[90]:


accuracy_rf = fit_rf.score(test_set, test_class_set)

print("Here is our mean accuracy on the test set:\n {0:.3f}"      .format(accuracy_rf))


# In[93]:


test_set


# In[94]:


test_class_set


# In[92]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(training_set, class_set)


# In[101]:


y_pred_lr=logreg.predict(test_set)


# In[102]:


y_pred_lr


# In[103]:


from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(test_class_set, y_pred_lr)
cnf_matrix


# In[104]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[105]:


print("Accuracy:",metrics.accuracy_score(test_class_set, y_pred_lr))
print("Precision:",metrics.precision_score(test_class_set, y_pred_lr))
print("Recall:",metrics.recall_score(test_class_set, y_pred_lr))


# In[107]:


import joblib

joblib.dump(fit_rf, 'diabetes_model.pkl')


# In[ ]:




