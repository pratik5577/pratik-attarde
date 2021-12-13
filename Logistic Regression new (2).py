#!/usr/bin/env python
# coding: utf-8

# # Output variable -> y
# y -> Whether the client has subscribed a term deposit or not 
# Binomial ("yes" or "no")
# 

# In[3]:


import pandas as pd
import seaborn as sb
from sklearn.linear_model import LogisticRegression


# Looking briefly at the data columns, we are can see that there are various numerical and categorical columns! These columns can be explained in more details below:

# In[144]:


df = pd.read_csv("bank-full.csv",sep=';')
df.tail()


# In[145]:


df.head()


# In[146]:


data.columns


# In[129]:


#count the number of rows for each type
df.groupby('y').size()


# In[5]:


# select columns
columns = ['age', 'balance', 'duration', 'campaign', 'y']
df_sel = df[columns]
df_sel.info()


# In[28]:


df.groupby('y').mean()


# In[29]:


df.groupby('job').mean()


# In[30]:


df.groupby('marital').mean()


# In[31]:


df.groupby('education').mean()


# The most important column here is y, which is the output variable (desired target): this will tell us if the client subscribed to a term deposit(binary: ‘yes’,’no’).

# In[6]:


get_ipython().system('pip install pandas matplotlib')


# # Visualizations

# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df.job,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')


# The frequency of purchase of the deposit depends a great deal on the job title. Thus, the job title can be a good predictor of the outcome variable.

# In[35]:


table=pd.crosstab(df.marital,df.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')


# The marital status does not seem a strong predictor for the outcome variable.

# In[36]:


table=pd.crosstab(df.education,df.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')


# Education seems a good predictor of the outcome variable.

# In[38]:


pd.crosstab(df.day,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day ')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')


# Day  may not be a good predictor of the outcome

# In[39]:


pd.crosstab(df.month,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar')


# Month might be a good predictor of the outcome variable.

# In[40]:


d.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')


# Most of the customers of the bank in this dataset are in the age range of 30–40.

# In[41]:


pd.crosstab(df.poutcome,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_pout_bar')


# Poutcome seems to be a good predictor of the outcome variable.

# In[ ]:





# define a function in order to calculate the prevalence of population that subscribes to a term deposit.

# In[7]:


pd.crosstab(df_sel.age,df_sel.y).plot(kind="line")


# In[8]:


sb.boxplot(data =df_sel,orient = "v")


# In[9]:


df_sel['outcome'] = df_sel.y.map({'no':0, 'yes':1})
df_sel.tail(10)


# In[10]:


df_sel.boxplot(column='age', by='outcome')


# #probably not a great feature since lot of outliers

# In[11]:


feature_col=['age','balance','duration','campaign']
output_target=['outcome']
X = df_sel[feature_col]
Y = df_sel[output_target]


# In[18]:


classifier = LogisticRegression()


# In[19]:


classifier.fit(X,Y)


# In[20]:


classifier.coef_ # coefficients of features 


# In[21]:


classifier.predict_proba (X) # Probability values


# In[22]:


y_pred = classifier.predict(X)


# In[23]:


y_pred


# In[24]:


from sklearn.metrics import confusion_matrix


# In[25]:


confusion_matrix = confusion_matrix(Y,y_pred)


# In[26]:


print (confusion_matrix)


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


sb.heatmap(confusion_matrix, annot=True)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')


# # ROC curve on the dataset

# In[114]:


from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# In[115]:


breast_cancer = load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target


# In[116]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=44)


# In[117]:


clf = LogisticRegression(penalty='l2', C=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[118]:


print("Accuracy", metrics.accuracy_score(y_test, y_pred))


# In[119]:


y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[73]:


pip install plot-metric 


# In[78]:


pip install metriculous


# In[82]:


#define the predictor variables and the response variable
X = df[['age', 'balance', 'campaign']]
y = data['default']


# In[83]:


#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 


# In[84]:


#instantiate the model
log_regression = LogisticRegression()


# In[85]:


#fit the model using the training data
log_regression.fit(X_train,y_train)


# In[120]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)


# In[121]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, random_state=23)
model = clf.fit(X_train, y_train)

# Use predict_proba to predict probability of the class
y_pred = clf.predict_proba(X_test)[:,1]


# In[122]:


from plot_metric.functions import BinaryClassification
# Visualisation with plot_metric
bc = BinaryClassification(y_test, y_pred, labels=["Class 1", "Class 2"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()


# In[61]:


import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
fpr

tpr

thresholds


# In[105]:


import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)


# In[ ]:




