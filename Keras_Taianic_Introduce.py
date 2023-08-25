#!/usr/bin/env python
# coding: utf-8

# # 下載鐵達尼號旅客資料集

# In[1]:


import urllib.request
import os


# In[2]:


url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath="data/titanic3.xls"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)


# # 使用Pandas dataframe讀取資料並進行處理

# In[3]:


import numpy
import pandas as pd


# In[4]:


all_df = pd.read_excel(filepath)


# In[5]:


all_df[:2]


# In[6]:


cols=['survived','name','pclass' ,'sex', 'age', 'sibsp',
      'parch', 'fare', 'embarked']
all_df=all_df[cols]


# In[7]:


all_df[:2]


# In[8]:


all_df.isnull().sum()


# In[9]:


df=all_df.drop(['name'], axis=1)


# In[10]:


age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)


# In[11]:


fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)


# In[12]:


df['sex']= df['sex'].map({'female':0, 'male': 1}).astype(int)


# In[13]:


df[:2]


# In[14]:


x_OneHot_df = pd.get_dummies(data=df,columns=["embarked" ])


# In[15]:


x_OneHot_df[:2]


# # 轉換為array

# In[16]:


ndarray = x_OneHot_df.values


# In[17]:


ndarray.shape


# In[18]:


ndarray[:2]


# In[19]:


Label = ndarray[:,0]
Features = ndarray[:,1:]


# In[ ]:





# In[20]:


Features.shape


# In[21]:


Features[:2]


# In[22]:


Label.shape


# In[23]:


Label[:2]


# # 將array進行標準化

# In[24]:


from sklearn import preprocessing


# In[25]:


minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))


# In[26]:


scaledFeatures=minmax_scale.fit_transform(Features)


# In[27]:


scaledFeatures[:2]


# In[28]:


Label[:5]


# # 將資料分為訓練資料與測試資料

# In[29]:


msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]


# In[30]:


print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))


# In[31]:


def PreprocessData(raw_df):
    df=raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex']= df['sex'].map({'female':0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df,columns=["embarked" ])

    ndarray = x_OneHot_df.values
    Features = ndarray[:,1:]
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label


# In[32]:


train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)


# In[33]:


train_Features[:2]


# In[34]:


train_Label[:2]

