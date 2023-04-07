#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Insurance_prediction.csv")


# In[3]:


df.head()


# In[4]:


df.drop(columns = ['_id'], inplace = True)


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.isna().sum()


# In[9]:


df.duplicated()


# In[10]:


df.describe()


# In[11]:


df.shape


# #EDA

# In[12]:


numerical_feature = [feature for feature in df.columns if df[feature].dtype != 'O']
print(f"{numerical_feature}")


# In[13]:


catagorical_feature = [feature for feature in df.columns if df[feature].dtype == 'O']
print(f"{catagorical_feature}")


# In[14]:


for col in catagorical_feature:
    print(df[col].value_counts(normalize = True)*100)
    print('--'*50)


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.suptitle("Univariate Analysis of Numerical Features", fontsize = 20, fontweight = 'bold')

for i in range(0,len(numerical_feature)):
    plt.subplot(2,2,i+1)
    sns.kdeplot(x = df[numerical_feature[i]], shade = True, color = 'b')
    plt.xlabel(numerical_feature[i])
    plt.tight_layout()
    
plt.savefig("Univariate_Num.png")


# In[16]:


df[numerical_feature].corr


# In[17]:


plt.figure(figsize=(30,25))
sns.heatmap(df.corr(), annot = True)


# # Outliers and Histogram

# In[18]:


fig, ax = plt.subplots(7,2, figsize=(20,65))
idx = 0

for i, column in enumerate (numerical_feature):
    sns.histplot(data=df, x=column, fill=True, kde = True, ax=ax[idx][0]).set_title(f'Distribution of {column}', fontsize='12')
    sns.boxplot(data=df, x=column, orient='h', ax=ax[idx][1]).set_title(f'Box Plot of {column}', fontsize='12')
    idx +=1

plt.show()


# In[19]:


import dtale
d = dtale.show(df)
d.open_browser()


# In[ ]:





# #Univariate analysis of categorical variable

# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.suptitle("Univariate Analysis of Categorical Features", fontsize = 20, fontweight = 'bold')

for i in range(0,len(catagorical_feature)):
    plt.subplot(2,2,i+1)
    sns.countplot(data = df, x = df[catagorical_feature[i]])
    plt.xlabel(catagorical_feature[i])
    plt.tight_layout()
    
plt.savefig("Univariate_Cat.png")


# #Bivariate Analysis

# Scatterplot (Numerical - Numerical)

# In[21]:


sns.scatterplot(df['age'],df['bmi'], hue = df['expenses'])


# #Barplot (Numerical - Categorical)

# In[22]:


sns.barplot(df['sex'],df['age'], hue = df['smoker'])


# #Boxplot (Numerical - Categorical)

# In[23]:


sns.boxplot(df['sex'],df['age'], hue = df['region'])


# In[24]:


#Multivariate analysis


# In[25]:


sns.pairplot(df)


# In[26]:


#Gender column
plt.figure(figsize=(8,8))
sns.countplot(x='sex', data= df)
plt.title('Sex Distribution')
plt.show()


# In[27]:


#bmi distribution
plt.figure(figsize=(8,8))
sns.distplot(df['bmi'], color='green')
plt.title('BMI Distribution')
plt.show()


# In[28]:


#children field distribution
plt.figure(figsize=(8,8))
sns.countplot(x='children', data=df)
plt.title('Children Count Distribution')
plt.show()


# In[29]:


plt.figure(figsize=(8,8))
sns.countplot(x='smoker', data=df)
plt.title('smoker Count Distribution')
plt.show()


# In[30]:


#region field distribution
plt.figure(figsize=(8,8))
sns.countplot(x='region', data=df)
plt.title('Region Count Distribution')
plt.show()


# In[31]:


#charges distribution
plt.figure(figsize=(8,8))
sns.distplot(df['expenses'], color='red')
plt.title('Expenses Distribution')
plt.show()


# In[32]:


X = df.drop(columns = 'expenses', axis =1)
Y = df['expenses']


# In[33]:


X


# In[34]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler


# In[35]:


num_features = X.select_dtypes(exclude = 'object').columns
cat_features = X.select_dtypes(include = 'object').columns


# In[36]:


num_features


# In[37]:


cat_features


# In[38]:


from sklearn.compose import ColumnTransformer
cat_transformer = OneHotEncoder()
num_transformer = StandardScaler()

preprocessor = ColumnTransformer (

[    
    ('OneHotEncoding',cat_transformer,cat_features),
    ('StandardScaling',num_transformer,num_features)  
]
)


# In[39]:


X_new = pd.DataFrame(preprocessor.fit_transform(X))


# In[40]:


X_new


# In[41]:


X.keys()


# In[42]:


X = df.drop(columns = 'expenses', axis =1)
Y = df['expenses']


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, random_state=2)


# # Linear Regression

# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


simple_model = LinearRegression()
simple_model.fit(X_train, Y_train)


# In[46]:


def model_report(y_test, Y_pred):
    print(f"Accuracy: {simple_model.score(X_test, Y_test) * 100:.2f}%")
    print(f"MAE: {mean_absolute_error(Y_test, Y_pred)}")
    print(f"MSE: {mean_squared_error(Y_test, Y_pred)}")
    print(f"R2: {r2_score(Y_test, Y_pred)}")


# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

Y_pred = simple_model.predict(X_test)

model_report(Y_test, Y_pred)


# # Random Forest

# In[52]:



from sklearn.ensemble import RandomForestRegressor

def model_report(y_test, Y_pred):
    print(f"Accuracy: {RFR.score(X_test, Y_test) * 100:.2f}%")
    print(f"MAE: {mean_absolute_error(Y_test, Y_pred)}")
    print(f"MSE: {mean_squared_error(Y_test, Y_pred)}")
    print(f"R2: {r2_score(Y_test, Y_pred)}")


RFR = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
RFR.fit(X_train,Y_train)
Y_pred = RFR.predict(X_test)
model_report(Y_test,Y_pred)


# # Support Vector Machine

# In[51]:


from sklearn.svm import SVR
def model_report(y_test, Y_pred):
    print(f"Accuracy: {SVC.score(X_test, Y_test) * 100:.2f}%")
    print(f"MAE: {mean_absolute_error(Y_test, Y_pred)}")
    print(f"MSE: {mean_squared_error(Y_test, Y_pred)}")
    print(f"R2: {r2_score(Y_test, Y_pred)}")



SVC= SVR(kernel='rbf')
SVC.fit(X_train,Y_train)
Y_pred = SVC.predict(X_test)
model_report(Y_test,Y_pred)

