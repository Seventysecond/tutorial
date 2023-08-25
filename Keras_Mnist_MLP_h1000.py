#!/usr/bin/env python
# coding: utf-8

# # 資料預處理

# In[1]:


from keras.utils import np_utils
import numpy as np
np.random.seed(10)


# In[2]:


from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label)= mnist.load_data()


# In[3]:


x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')


# In[4]:


x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255


# In[5]:


y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)


# # 建立模型

# In[6]:


from keras.models import Sequential
from keras.layers import Dense


# In[7]:


model = Sequential()


# In[8]:


#將「輸入層」與「隱藏層」加入模型


# In[9]:


model.add(Dense(units=1000, 
                input_dim=784, 
                kernel_initializer='normal', 
                activation='relu'))


# In[10]:


#將「輸出層」加入模型


# In[11]:


model.add(Dense(units=10, 
                kernel_initializer='normal', 
                activation='softmax'))


# In[12]:


print(model.summary())


# # 訓練模型

# In[13]:


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


# In[14]:


train_history=model.fit(x=x_Train_normalize,
                        y=y_Train_OneHot,validation_split=0.2, 
                        epochs=10, batch_size=200,verbose=2)


# # 以圖形顯示訓練過程

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


scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy=',scores[1])


# # 進行預測

# In[19]:


prediction=model.predict_classes(x_Test)


# In[20]:


prediction


# In[21]:


import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" +str(labels[idx])+
                     ",predict="+str(prediction[idx])
                     ,fontsize=10) 
        
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[22]:


plot_images_labels_prediction(x_test_image,y_test_label,
                              prediction,idx=340)


# # confusion matrix

# In[23]:


import pandas as pd
pd.crosstab(y_test_label,prediction,
            rownames=['label'],colnames=['predict'])


# In[24]:


df = pd.DataFrame({'label':y_test_label, 'predict':prediction})
df[:2]


# In[25]:


df[(df.label==5)&(df.predict==3)]


# In[26]:


plot_images_labels_prediction(x_test_image,y_test_label
                              ,prediction,idx=340,num=1)


# In[27]:


plot_images_labels_prediction(x_test_image,y_test_label
                              ,prediction,idx=1289,num=1)

