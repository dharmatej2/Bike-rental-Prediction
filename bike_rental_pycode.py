
# coding: utf-8

# In[1]:

#importing required packages for initial data analysis

import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

#Set working directory
os.chdir("C:\\Users\\Sunny\\Desktop\\edwisor\\Bike rental")
os.getcwd()


# In[85]:

#reading the data

bike = pd.read_csv("day.csv")

#data sample
bike.head(5)


# In[18]:

#dimensions of the data
bike.shape


# In[12]:

#description of data
bike.describe()


# In[16]:

#data types of individual variables
bike.dtypes


# In[20]:

#counting no.of observations of each variable
bike.count()
#all coulumns have the same no.of rows indicting there are no missing values


# In[21]:

#also Finding the missing values in a more efficiengt way.
print(bike.isnull().sum())
#no missing values


# In[23]:

#converting required data types
bike['season']= bike['season'].astype('category')
bike['yr']=bike['yr'].astype('category')
bike['mnth']=bike['mnth'].astype('category')
bike['holiday']=bike['holiday'].astype('category')
bike['workingday']=bike['workingday'].astype('category')
bike['weekday']=bike['weekday'].astype('category')
bike['weathersit']=bike['weathersit'].astype('category')
bike.dtypes


# In[41]:

#outlier analysis

#boxplot for temp variable 
plt.boxplot(bike['temp'], showfliers=True)

#sns.set(style="whitegrid")
#sns.boxplot(x=bike['temp'],orient ='h')
sns.plt.show()


# In[42]:

#boxplot for atemp
plt.boxplot(bike['atemp'], showfliers=True)

sns.plt.show()


# In[43]:

#boxplot for humidity
plt.boxplot(bike['hum'], showfliers=True)

sns.plt.show()


# In[44]:

#boxplot for windspeed
plt.boxplot(bike['windspeed'], showfliers=True)

sns.plt.show()


# In[50]:

#we dont see any outliers present in the normalized numeric variables
#lets find correlation between variables

bike_corr = bike

f, ax = plt.subplots(figsize=(10,7))

#Generate correlation matrix
corr = bike_corr.corr()
#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
sns.plt.show()



# In[54]:

#we see that temp and atemp are highly correlated
#hence dropping atemp variable from data
bike = bike.drop(['atemp'], axis=1)


# In[56]:

bike.shape


# In[57]:

#also dropping other unnecessary variables ,like instant, dteday, casual and registered
#casual + registered = cnt
bike = bike.drop(['instant','dteday','casual', 'registered'], axis=1)


# In[59]:

bike.shape
bike.describe()


# In[62]:

bike.dtypes


# In[ ]:

#importing required packages for model building

from scipy.stats import chi2_contingency
from random import randrange, uniform
import datetime as dt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor



# In[63]:

#model building
#linear regression

bike_lr=bike.copy()
bike.shape


# In[64]:

bike_lr.shape


# In[65]:

cat_names = ["season","yr","holiday","workingday","weathersit", "mnth","weekday"]
cat_names


# In[66]:

#dummify

for i in cat_names:
    temp = pd.get_dummies(bike_lr[i], prefix = i)
    bike_lr = bike_lr.join(temp)
bike_lr.shape


# In[67]:

fields_to_drop = ['yr', 'season','holiday','workingday', 'weathersit', 'weekday', 'mnth','cnt']
bike_lr = bike_lr.drop(fields_to_drop, axis=1)
bike_lr.shape


# In[68]:

bike_lr=bike_lr.join(bike['cnt'])

bike_lr.shape


# In[71]:

#splitting data in to train and test
bike_trainlr, bike_testlr = train_test_split(bike_lr, test_size=0.2)

#building logistic regression
LR_model = sm.OLS(bike_trainlr.iloc[:,35], bike_trainlr.iloc[:,0:35]).fit()

#predicting values using logisting regression model
predictions_LR = LR_model.predict(bike_testlr.iloc[:,0:35])


# In[87]:

LR_model.summary()


# In[77]:

#dividing data into train and test
train, test = train_test_split(bike, test_size=0.2)



# In[78]:

#building Decision tree model
DT_model = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:10], train.iloc[:,10])

#predicting values using decision tree model
predictions_DT = DT_model.predict(test.iloc[:,0:10])



# In[79]:

#building random forest
RF_model = RandomForestRegressor(n_estimators = 200).fit(train.iloc[:,0:10], train.iloc[:,10])

#predicting values using random forest model
RF_Predictions = RF_model.predict(test.iloc[:,0:10])


# In[81]:

#defining MAPE function
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape



#MAPE for linear regression
MAPE(bike_testlr.iloc[:,35], predictions_LR)



# In[82]:

#MAPE for decision tree regression
MAPE(test.iloc[:,10], predictions_DT)



# In[83]:

#MAPE for random forest regression
MAPE(test.iloc[:,10],RF_Predictions)


# In[84]:

#writing predictited values to csv file
result=pd.DataFrame(test.iloc[:,0:11])
result['pred_cnt'] = (RF_Predictions)

result.to_csv("bike_rental_RF_python.csv",index=False)

