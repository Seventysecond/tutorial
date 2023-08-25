#!/usr/bin/env python
# coding: utf-8

# # 1. Import Library

# In[1]:


import numpy
import pandas as pd
from sklearn import preprocessing
numpy.random.seed(10)


# # 資料準備

# In[2]:


all_df = pd.read_excel("data/titanic3.xls")


# In[3]:


cols=['survived','name','pclass' ,'sex', 'age', 'sibsp',
      'parch', 'fare', 'embarked']
all_df=all_df[cols]


# In[4]:


msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]


# In[5]:


print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))


# In[6]:


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


# In[7]:


train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)


# # 3. Create Model 

# In[8]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[9]:


model = Sequential()


# In[10]:


model.add(Dense(units=40, input_dim=9, 
                kernel_initializer='uniform', 
                activation='relu'))


# In[11]:


model.add(Dense(units=30, 
                kernel_initializer='uniform', 
                activation='relu'))


# In[12]:


model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))


# # 4. Train model

# In[13]:


model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


# In[14]:


train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=30, 
                         batch_size=30,verbose=2)


# # 6. Print History

# In[15]:


import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[16]:


show_train_history(train_history,'acc','val_acc')


# In[17]:


show_train_history(train_history,'loss','val_loss')


# # 評估模型準確率

# In[18]:


scores = model.evaluate(x=test_Features, 
                        y=test_Label)


# In[19]:


scores[1]


# # 預測資料

# # 加入Jack & Rose資料

# In[20]:


Jack = pd.Series([0 ,'Jack',3, 'male'  , 23, 1, 0,  5.0000,'S'])
Rose = pd.Series([1 ,'Rose',1, 'female', 20, 1, 0, 100.0000,'S'])


# In[21]:


JR_df = pd.DataFrame([list(Jack),list(Rose)],  
                  columns=['survived', 'name','pclass', 'sex', 
                   'age', 'sibsp','parch', 'fare','embarked'])


# In[22]:


all_df=pd.concat([all_df,JR_df])


# In[23]:


all_df[-2:]


# # 進行預測

# In[24]:


all_Features,Label=PreprocessData(all_df)


# In[25]:


all_probability=model.predict(all_Features)


# In[26]:


all_probability[:10]


# In[27]:


pd=all_df
pd.insert(len(all_df.columns),
          'probability',all_probability)


# # 預測Jack & Rose資料的生存機率

# In[28]:


pd[-2:]


# # 查看生存機率高，卻沒有存活

# In[29]:


pd[(pd['survived']==0) &  (pd['probability']>0.9) ]


# In[30]:


pd[:5]

