#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.tree import DecisionTreeClassifier


# In[4]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# In[5]:


from sklearn.ensemble import RandomForestClassifier


# In[6]:


from sklearn.naive_bayes import GaussianNB
bayesmodel = GaussianNB()


# In[7]:


treemodel=DecisionTreeClassifier(ccp_alpha=0.01)


# In[8]:


dataset = pd.read_csv (r'C:\Users\alelawson\OneDrive - Deloitte (O365D)\Desktop\Headshot\Sklearn\train.csv') 


# In[9]:


testset = pd.read_csv (r'C:\Users\alelawson\OneDrive - Deloitte (O365D)\Desktop\Headshot\Sklearn\test.csv') 


# In[10]:


xp=dataset.drop(['Name','Ticket','Fare','Cabin','Embarked','PassengerId','Survived'], axis=1)


# In[11]:


#Make all the variables numeric

dataset["Sex"].replace({"male": "1","female": "2"}, inplace=True)
testset["Sex"].replace({"male": "1","female": "2"}, inplace=True)


# In[12]:


#make the training dataset

y=dataset['Survived']
X=dataset.drop(['Name','Ticket','Fare','Cabin','Embarked','PassengerId','Survived'], axis=1)


# In[13]:



# we are splitting the data to make one trained set (80% of the data) and one test set (20% of the data) 
#80% if the data is training the model, 20% is unseen

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=0)


# In[14]:


#Now we 80% if the data is training the model, 20% is unseen


treemodel.fit(X_train,y_train)


# In[15]:


treemodel.feature_importances_


# In[16]:


features = ['Pclass', 'Sex', 'Age','SibSp','Parch']
classes = ['Deceased', 'Survived']

plt.figure(figsize=(10, 4))
plot_tree(treemodel, feature_names=features, class_names=classes, filled=True);


# # Testing the decision tree model against the test data

# In[17]:


t_test=testset.drop(['Name','Ticket','Fare','Cabin','Embarked','PassengerId'], axis=1)


# In[18]:


p=treemodel.predict(t_test)


# In[19]:


df = pd.DataFrame (p, columns = ['prediciton'])
print (df)


# Running Random Forrest

# In[20]:


rf_result=rf.predict(t_test)
rf_result = pd.DataFrame (rf_result, columns = ['prediciton'])
print (rf_result)


# In[21]:


rf = RandomForestClassifier(n_estimators=10, criterion='entropy')
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)


# # Fitting the bayes model 

# In[22]:


bayesmodel.fit(X_train,y_train)


# In[23]:


X_train 


# In[71]:


bayesmodel.score(X_test,y_test)


# In[63]:


#testing the random forrest (rf)

print("Training Accuracy is: ", rf.score(X_train, y_train))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_test, y_test))


# In[64]:


#testing the decision tree


# Accuracy on Train
print("Training Accuracy is: ", treemodel.score(X_train, y_train))

# Accuracy on Train
print("Testing Accuracy is: ", treemodel.score(X_test, y_test))


# In[65]:


result = pd.concat([rf_result, df], axis=1, join='inner')
display(result)


# In[66]:


rf_result.to_csv(r'C:\Users\alelawson\OneDrive - Deloitte (O365D)\Desktop\Headshot\Sklearn\Prediction output.csv')


# In[ ]:





# In[ ]:




