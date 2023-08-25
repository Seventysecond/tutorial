#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
width = tf.placeholder("int32",name='width')
height = tf.placeholder("int32",name='height')
area=tf.multiply(width,height,name='area')  

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area=',sess.run(area,feed_dict={width: 6,height: 8}))


# In[2]:


tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/area',sess.graph)

