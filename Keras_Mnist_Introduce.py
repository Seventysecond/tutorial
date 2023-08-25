#!/usr/bin/env python
# coding: utf-8

# # 資料準備

# In[ ]:


import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)


# In[ ]:


from keras.datasets import mnist


# In[ ]:


(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()


# In[ ]:


print('train data=',len(x_train_image))
print(' test data=',len(x_test_image))


# In[ ]:


print ('x_train_image:',x_train_image.shape)
print ('y_train_label:',y_train_label.shape)


# In[ ]:


import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()


# In[ ]:


plot_image(x_train_image[0])


# In[ ]:


y_train_label[0]


# In[ ]:


import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[ ]:


plot_images_labels_prediction(x_train_image,y_train_label,[],0,10)


# In[ ]:


print ('x_test_image:',x_test_image.shape)
print ('y_test_label:',y_test_label.shape)


# In[ ]:


plot_images_labels_prediction(x_test_image,y_test_label,[],0,10)


# # 將images進行預處理

# In[ ]:


print ('x_train_image:',x_train_image.shape)
print ('y_train_label:',y_train_label.shape)


# In[ ]:


x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')


# In[ ]:


print ('x_train:',x_Train.shape)
print ('x_test:',x_Test.shape)


# In[ ]:


x_train_image[0]


# In[ ]:


x_Train_normalize = x_Train/ 255
x_Test_normalize = x_Test/ 255


# In[ ]:


x_Train_normalize[0]


# # one hot encode outputs

# In[ ]:


y_train_label[:5]


# In[ ]:


y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)


# In[ ]:


y_TrainOneHot[:5]

