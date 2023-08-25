#!/usr/bin/env python
# coding: utf-8

# # 5.1	建立Computational Graph

# # 匯入tensorflow模組

# In[1]:


import tensorflow as tf


# # 建立 const 

# In[2]:


ts_c = tf.constant(2,name='ts_c')


# In[3]:


ts_c


# # 建立 Variable

# In[4]:


ts_x = tf.Variable(ts_c+5,name='ts_x')


# In[5]:


ts_x


# # 5.2	建立Session執行Computational Graph

# In[6]:


sess=tf.Session()


# In[7]:


init = tf.global_variables_initializer()
sess.run(init)


# In[8]:


print('ts_c=',sess.run(ts_c))


# In[9]:


print('ts_x=',sess.run(ts_x))


# In[10]:


print('ts_c=',ts_c.eval(session=sess))


# In[11]:


print('ts_x=',ts_x.eval(session=sess))


# In[12]:


sess.close()    


# # Session open close

# In[13]:


import tensorflow as tf
ts_c = tf.constant(2,name='ts_c')
ts_x = tf.Variable(ts_c+5,name='ts_x')

sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print('ts_c=',sess.run(ts_c))
print('ts_x=',sess.run(ts_x))
sess.close()


# # With語法開啟Session

# In[14]:


import tensorflow as tf
ts_c = tf.constant(2,name='ts_c')
ts_x = tf.Variable(ts_c+5,name='ts_x')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('ts_c=',sess.run(ts_c))
    print('ts_x=',sess.run(ts_x))


# # 4.placeholder

# In[15]:


width = tf.placeholder("int32")
height = tf.placeholder("int32")
area=tf.multiply(width,height)


# In[16]:


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area=',sess.run(area,feed_dict={width: 6, height: 8}))


# # import

# In[ ]:





# # Create dim  1 tensor

# In[17]:


ts_X = tf.Variable([0.4,0.2,0.4])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X=sess.run(ts_X)
    print(X)


# In[18]:


print(X.shape)


# # Create dim  2 tensor

# In[19]:


ts_X = tf.Variable([[0.4,0.2,0.4]])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X=sess.run(ts_X)
    print(X)   


# In[20]:


print('shape:',X.shape)


# In[21]:


W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    W_array=sess.run(W)
    print(W_array)   
                        


# In[22]:


print(W_array.shape)


# # matmul 

# In[23]:


X = tf.Variable([[1.,1.,1.]])

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])
                        
XW =tf.matmul(X,W )
                       
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(XW ))


# # tf.add

# In[24]:


b = tf.Variable([[ 0.1,0.2]])
XW =tf.Variable([[-1.3,0.4]])

Sum =XW+b
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('Sum:')    
    print(sess.run(Sum ))


# # Y=X*W+b

# In[25]:


X = tf.Variable([[1.,1.,1.]])

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])
                         

b = tf.Variable([[0.1,0.2]])
    
XWb =tf.matmul(X,W)+b


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb:')    
    print(sess.run(XWb ))

