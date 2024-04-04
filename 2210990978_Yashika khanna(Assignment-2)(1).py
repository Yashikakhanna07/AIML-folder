#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data=pd.read_csv("C:\\Users\\Yashika\\OneDrive\\Desktop\\AI-ML\\insurance.csv")
data


# ### 1. Display Top 5 Rows of The Dataset

# In[2]:


data.head(5)


# ### 2. Check Last 5 Rows of The Dataset

# In[3]:


data.tail(5)


# ### 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[4]:


print ("Number of Rows" , data.shape[0])
print ("Number of Columns" , data.shape[1])


# ### 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[5]:


data.info()


# ### 5.Check Null Values In The Dataset

# In[6]:


data.isnull()


# ### 6. Get Overall Statistics About The Dataset

# In[7]:


data.describe()


# ### 7. Covert Columns From String ['sex' ,'smoker','region' ] To Numerical Values

# In[8]:


sex_mapping = {'male': 0, 'female': 1}
smoker_mapping = {'no': 0, 'yes': 1}
region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}

data['sex'] = data['sex'].replace(sex_mapping)
data['smoker'] = data['smoker'].replace(smoker_mapping)
data['region'] = data['region'].replace(region_mapping)
data


# ### 8. Store Feature Matrix In X and Response(Target) In Vector y

# In[9]:


X = data.drop('charges', axis=1)
y = data['charges']
X


# ### 9. Train/Test split

# 1. Split data into two part : a training set and a testing set
# 2. Train the model(s) on training set
# 3. Test the Model(s) on Testing set

# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train


# ### 10. Import the models

# In[11]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# ### 11. Model Training

# In[12]:


model.score(X_train, y_train)
model.score(X_test, y_test)


# ### 12. Prediction on Test Data

# In[24]:


y_pred = model.predict(X_test)
y_pred


# ### 13. Compare Performance Visually

# In[18]:


data = pd.DataFrame({'Actual' : y_test, 'Prediction' : y_pred})
data


# ### 14. Predict Charges For New Customer

# In[25]:


model.predict([[40,1,24.312,2,1,2]])

