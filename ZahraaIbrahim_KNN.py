#!/usr/bin/env python
# coding: utf-8

# ## Importing our libraries

# In[1]:


import pandas as pd # used for dataframe
import numpy as np # used for Maths
import matplotlib.pyplot as plt # used for plot the data
import seaborn as sns # used for plot the data 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler ## scale the data between 0 and 1
from sklearn.model_selection import train_test_split # split the data to (train & test) data
from sklearn.neighbors import KNeighborsClassifier # used to make classifications or predictions
from sklearn import metrics # used to assess the quality of your predictions.


# ## Importing our data

# In[2]:


dataset = pd.read_csv('Shill Bidding Dataset.csv')


# In[3]:


# we will review our data 
dataset


# In[4]:


# we want to know which (two cloumns ID unique)
len(dataset) == len(dataset['Record_ID'].unique())


# In[5]:


len(dataset) == len(dataset['Auction_ID'].unique())


# In[6]:


dataset.set_index('Record_ID', inplace=True)


# In[7]:


dataset['Bidder_ID'].value_counts()


# In[8]:


# dropping unnecssery columns  
dataset=dataset.drop(['Bidder_ID'],axis=1)


# In[9]:


# read the fist and last five columns 
dataset.head()


# In[10]:


dataset.tail()


# ## Getting information of the data

# In[11]:


dataset.shape


# In[12]:


dataset.size


# In[13]:


dataset.info()


# ## Checking for missing values

# In[14]:


dataset.describe(include= 'all')


# In[15]:


dataset[dataset.duplicated()].count()


# In[16]:


count= dataset.isnull().sum().sort_values(ascending=False)
percentage = ((dataset.isnull().sum()/len(dataset))*100).sort_values(ascending=False)
missing_data = pd.concat([count,percentage],axis=1,keys=['Count','Percentage'])
print ('Count and Percentage of missing values for the columns:')
missing_data


# In[17]:


dataset['Class'].value_counts()


# In[18]:


print ('Percentage for defult\n')
print (round(dataset.Class.value_counts(normalize=True)*100,2))
round (dataset.Class.value_counts(normalize=True)*100,2).plot(kind='bar')
plt.title('percentage Distribution by Class type')
sns.countplot(dataset['Class'])


# # feature selection

# In[19]:


X = dataset.drop('Class',axis=1)
y = dataset ['Class'] -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
stratify=y, random_state=99)


# In[20]:


from imblearn.over_sampling import RandomOverSampler
resample = RandomOverSampler(random_state=0)
X_train_oversampled , y_train_oversampled = resample.fit_resample(X_train,y_train)
sns.countplot(x=y_train_oversampled)


# In[21]:


from imblearn.under_sampling import RandomUnderSampler
resample = RandomUnderSampler(random_state=0)
X_train_undersampled , y_train_undersampled = resample.fit_resample(X_train,y_train)
sns.countplot(x=y_train_undersampled)


# In[22]:


from imblearn.over_sampling import SMOTE
resampler = SMOTE(random_state=0)
X_train_smote , y_train_smote = resampler.fit_resample(X_train,y_train)
sns.countplot(x=y_train_smote)


# ## split and scale the data

# In[23]:


features=['Auction_ID','Bidder_Tendency','Bidding_Ratio','Successive_Outbidding',
         'Last_Bidding','Auction_Bids','Starting_Price_Average','Early_Bidding','Winning_Ratio',
         'Auction_Duration']


# In[24]:


#separating out the features
X=dataset.loc[:, features].values


# In[25]:


X


# In[26]:


target=dataset.iloc[:,-1].values


# In[27]:


target


# In[28]:


X_train,X_test,target_train,target_test= train_test_split(X,target,test_size=0.3,random_state=0)


# In[29]:


print (X_train.shape)
print (target_test.shape)


# In[30]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[31]:


X_train


# ## starting our classification

# In[32]:


Classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
Classifier.fit(X_train,target_train)


# In[33]:


target_pred= Classifier.predict(X_test)
print (target_pred)


# In[34]:


print(target_test)


# In[35]:


acc=metrics.accuracy_score(target_test,target_pred)
print('accurancy:%.2f\n\n'%(acc))
cm=metrics.confusion_matrix(target_test,target_pred)
print('Confusion Matrix:')
print(cm,'\n\n')
print('................................................')
result=metrics.classification_report(target_test,target_pred)
print('Classification Report:\n')
print(result)


# In[36]:


ax = sns.heatmap(cm, cmap='flare',annot=True,fmt='d')
plt.xlabel('predicted class',fontsize=12)
plt.ylabel('confusion Matrix',fontsize=12)
plt.show()

