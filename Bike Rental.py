
# coding: utf-8

# In[1719]:


import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns


# In[1720]:


df = pd.read_csv('train.csv')


# In[1721]:


df.dtypes


# In[1722]:


df.head()


# In[1723]:


df.shape


# In[1724]:


df.humidity.unique()


# In[1725]:


# df2 = pd.read_csv('test.csv')
# df2.tail()


# In[1726]:


df.describe(include='all')


# In[1727]:


#Since date is not a ver viable entity for predicting the number of bikes sold, we'll only extract the month  from the date
new = df["datetime"].str.split(" ", n=1, expand=True)
df["Date"] = new[0]
df["Time"] = new[1]


# In[1728]:


df.drop(columns=["datetime"], inplace= True)


# In[1729]:


new = df["Date"].str.split("-", n=2, expand=True)
df["Month"] = new[1]


# In[1730]:


df.head()


# In[1731]:


df.drop(columns=["Date"], inplace= True)


# In[1732]:


#Extracting the value of the hour when the bike is rented. This would give us an idea regarding the part of the day when most of the bikes are rented - morning, noon, evening
new = df["Time"].str.split(":", n=2, expand=True)
df["Hour"] = new[0]
df.drop(columns=["Time"], inplace= True)


# In[1733]:


#one hot encoding of season and weather
df = pd.get_dummies(df, columns=['season', 'weather'], drop_first=True)


# In[1734]:


df.head()


# In[1735]:


df = df.rename(columns={'season_2':'season_summer', 'season_3':'season_fall', 'season_4':'season_winter'})


# ## Correlation

# In[1737]:


#Calculating the correlation between probable correlated variables
a = np.asarray(df['count'])
b = np.asarray(df['windspeed'])


# In[1738]:


corr, p_value = pearsonr(a,b)
print(corr)
print(p_value)


# In[1739]:


#Since temp and atemp are highly correlated we make another column 'temp' having the value of (temp+atemp)/2 and drop the original temp and atemp
df['temp']= (df['temp'] + df['atemp'])/2


# In[1740]:


df.drop(columns=['atemp'], inplace=True)


# In[1741]:


df.windspeed.unique()


# In[1742]:


df.drop(columns=['casual','registered'], inplace=True)


# In[1743]:


df['Month'] = df['Month'].astype('int')
df['Hour'] = df['Hour'].astype('int')


# In[1744]:


for col in ['holiday', 'workingday', 'season_summer', 'season_fall','season_winter', 'weather_2', 'weather_3', 'weather_4']:
    df[col] = df[col].astype('category')


# In[1745]:


df['Hour']


# ## Visualization

# In[1746]:


sns.countplot(x='Month', data=df)
#Uniformly distributed over months


# In[1747]:


sns.countplot(x='Hour', data=df)
#uniformly distributed over hour of the day


# In[1748]:


# df.loc[ df['Month'] <= 4, 'Month'] = 0
# df.loc[(df['Month'] > 4) & (df['Month'] <= 8), 'Month'] = 1
# df.loc[ df['Month'] > 8, 'Month'] = 2


# In[1749]:


# df.loc[ df['Hour'] <= 5, 'Hour'] = 0
# df.loc[(df['Hour'] > 5) & (df['Hour'] <= 11), 'Hour'] = 1
# df.loc[(df['Hour'] > 11) & (df['Hour'] <= 17), 'Hour'] = 2
# df.loc[ df['Hour'] > 17, 'Hour'] = 3


# In[1750]:


df.head()


# In[1751]:


df = pd.get_dummies(df, columns=['Month'], drop_first=True)
#df = pd.get_dummies(df, columns=['Hour'], drop_first=True)


# In[1752]:


df.dtypes


# In[1753]:


df.head()


# In[1754]:


#sns.countplot(x='Hour', data=df)


# In[1755]:


#sns.countplot(x='Month', data=df)


# In[1756]:


#df = df.drop(columns=['holiday'])


# In[1757]:


df.head()


# ## Data scaling

# In[1759]:


import math
#taking log of dependent variable
df['count'] = df['count'].transform(lambda x: math.log(x))


# In[1760]:


df.describe()


# In[1761]:


# df['temp'] = df['temp']/42


# In[1762]:


# df['humidity'] = df['humidity']/100
# df['windspeed'] = df['windspeed']/56


# In[1763]:


df.describe()


# In[1764]:


y = df['count']


# In[1765]:


X = df.drop(columns=['count'])


# ## Model

# In[1766]:


#train_test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2)


# In[1767]:


y_train.shape


# #### Linear Regression

# In[1768]:


from sklearn.linear_model import LinearRegression


# In[1769]:


lreg = LinearRegression()
lreg.fit(X_train, y_train)


# In[1770]:


y_pred = lreg.predict(X_test)


# In[1771]:


# y_pred


# In[1772]:


from sklearn.metrics import mean_squared_error, r2_score
print((r2_score(y_test, y_pred)))


# In[1773]:


accuracy = lreg.score(X_test,y_test)
print(accuracy*100,'%')


# In[1774]:


# from sklearn.tree import DecisionTreeClassifier
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, y_train)
# y_pred = decision_tree.predict(X_test)


# In[1775]:


#print(accuracy_score(y_test, y_pred))


# In[1776]:


# print(rmsle(y_test, np.exp(y_pred)))


# In[1777]:


# from sklearn.svm import SVC


# In[1778]:


# svc = SVC()
# svc.fit(X_train, y_train)
# pred = svc.predict(X_test)
# svc.score(train_X, train_y)


# In[1779]:


def rmsle(prediction, actual):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in prediction]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in actual]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# #### Random Forest

# In[1780]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor().fit(X_train, y_train)
prediction_rfr = rfr.predict(X_test)


# In[1781]:


rmsle(prediction_rfr,y_test)


# In[1782]:


# print(r2_score(y_test, prediction_rfr))


# In[1783]:


accuracy = rfr.score(X_test,y_test)
print(accuracy*100,'%')


# #### Ensemble - Boosting methods

# In[1784]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

models = [LinearRegression(),
          Ridge(),
          HuberRegressor(),
          ElasticNetCV(),
          DecisionTreeRegressor(), 
          ExtraTreesRegressor(),
          GradientBoostingRegressor(),
          RandomForestRegressor(),
          BaggingRegressor()]

def test_algorithms(model):
    kfold = model_selection.KFold(n_splits=10, random_state=0)
    predicted = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print(predicted.mean())
    
for model in models:
    test_algorithms(model)

