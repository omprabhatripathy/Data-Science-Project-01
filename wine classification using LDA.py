#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
   
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 


# In[2]:


data = pd.read_csv("Wine.csv")


# In[3]:


data


# In[4]:


X= data.iloc[:, :-1 ]
X


# In[5]:


y = data.iloc[:,-1]
y


# In[6]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[7]:


X_train


# In[8]:


y_train


# In[9]:


X_test


# In[10]:


#feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[11]:


X_train


# In[12]:


#apply lda

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[13]:


lda = LDA(n_components =2)


# In[14]:


X_train = lda.fit_transform(X_train, y_train)


# In[15]:


X_test = lda.transform(X_test)


# In[16]:


#logistic regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)


# In[17]:


y_test


# In[18]:


y_pred


# In[19]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[20]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


###for visualising training set data 


# In[23]:


from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2 = np.meshgrid(
    np.arange(start = X_set[:,0].min()-1, stop=X_set[:,0].max() +1,step=0.25),
    np.arange(start = X_set[:,1].min()-1, stop=X_set[:,1].max() +1,step=0.25),

)


# In[24]:


X1


# In[29]:


plt.contourf(X1,X2,lr.predict(np.array([X1.ravel(),X2.ravel()]).T).
    reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','blue','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())


# In[30]:


for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j ,0],X_set[y_set== j,1],
               c= ListedColormap(('red','blue','green'))(i),label=j)


# In[32]:


plt.title("Training Set")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend
plt.show()


# In[ ]:


#combining the above into 1


# In[34]:


from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2 = np.meshgrid(
    np.arange(start = X_set[:,0].min()-1, stop=X_set[:,0].max() +1,step=0.25),
    np.arange(start = X_set[:,1].min()-1, stop=X_set[:,1].max() +1,step=0.25),

)
plt.contourf(X1,X2,lr.predict(np.array([X1.ravel(),X2.ravel()]).T).
    reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','blue','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j ,0],X_set[y_set== j,1],
               c= ListedColormap(('red','blue','green'))(i),label=j)

    
plt.title("Training Set")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()


# In[35]:


###for visualising testing set data 


# In[36]:


from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2 = np.meshgrid(
    np.arange(start = X_set[:,0].min()-1, stop=X_set[:,0].max() +1,step=0.25),
    np.arange(start = X_set[:,1].min()-1, stop=X_set[:,1].max() +1,step=0.25),

)
plt.contourf(X1,X2,lr.predict(np.array([X1.ravel(),X2.ravel()]).T).
    reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','blue','green')))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j ,0],X_set[y_set== j,1],
               c= ListedColormap(('red','blue','green'))(i),label=j)

    
plt.title("Test Set")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()


# In[ ]:




