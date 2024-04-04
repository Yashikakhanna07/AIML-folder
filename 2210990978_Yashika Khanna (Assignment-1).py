#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("C:\\Users\\yashika\\OneDrive\\Desktop\\AI-ML\\ST-2 Supervised learning algorithm\\add.csv")
df


# In[2]:


import matplotlib.pyplot as plt
plt.scatter(df['x'], df['sum'])


# In[3]:


import matplotlib.pyplot as plt
plt.scatter(df['y'], df['sum'])


# ### 1. Store feature matrix in X and response (target) in vector Y

# In[4]:


x = df[['x', 'y']]
y = df['sum']

x


y


# ### Train / Test Split

# 1. Split data into 2 part :- a training set & a testing set
# 2. Train the model on training set
# 3. Train the model on testing set

# In[8]:


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(
x, y , test_size = 0.33, random_state = 8)

y_test
x_train


# ### Import and Train the Model

# In[12]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

# when we are creating the model, we use fit method


# ### Check Model's Prediction Performance

# In[14]:


model.score(x_train, y_train)
model.score(x_test, y_test) 

# we use score method to predict the performance of training and testing variables


# ### Comparing The Results

# In[16]:


y_pred = model.predict(x_test)
y_pred

df = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
df


# ### Prediction

# In[19]:


model.predict([[10,20]])
model.predict([[100.2,210.3]])

