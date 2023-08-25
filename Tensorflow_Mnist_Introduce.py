#!/usr/bin/env python
# coding: utf-8

# # 下載並讀取資料

# In[1]:


import tensorflow as tf


# In[2]:


import tensorflow.examples.tutorials.mnist.input_data as input_data


# In[3]:


#first time


# In[3]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[5]:


#讀取資料


# In[4]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[7]:


print('train',mnist.train.num_examples,
      ',validation',mnist.validation.num_examples,
      ',test',mnist.test.num_examples)


# # 查看train Data

# In[8]:


print('train images     :', mnist.train.images.shape,
      'labels:'           , mnist.train.labels.shape)


# In[9]:


len(mnist.train.images[0])


# In[10]:


mnist.train.images[0]


# In[11]:


import matplotlib.pyplot as plt
def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='binary')
    plt.show()


# In[12]:


plot_image(mnist.train.images[0])


# In[13]:


mnist.train.labels[0]


# In[14]:


import numpy as np
np.argmax(mnist.train.labels[0])


# In[15]:


for i in range(10):
    print(mnist.train.labels[i])


# In[37]:


import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        
        ax.imshow(np.reshape(images[idx],(28, 28)), 
                  cmap='binary')
            
        title= "label=" +str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


# In[38]:


plot_images_labels_prediction(mnist.train.images,
                              mnist.train.labels,[],0)


# # read validation data

# In[19]:


print('validation images:', mnist.validation.images.shape,
      'labels:'           , mnist.validation.labels.shape)


# In[39]:


plot_images_labels_prediction(mnist.validation.images,
                              mnist.validation.labels,[],0)


# # read test data

# In[21]:


print('test images:', mnist.test.images.shape,
      'labels:'           , mnist.test.labels.shape)


# In[40]:


plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels,[],0)


# # read batch

# In[27]:


batch_images_xs, batch_labels_ys =      mnist.train.next_batch(batch_size=100)


# In[24]:


print(len(batch_images_xs),
      len(batch_labels_ys))


# In[41]:


plot_images_labels_prediction(batch_images_xs,
                              batch_labels_ys,[],0)

