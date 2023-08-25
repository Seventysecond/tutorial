#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# # y= relu ( (X․W ) + b )

# In[2]:


X = tf.Variable([[0.4,0.2,0.4]])

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])
                         
b = tf.Variable([[0.1,0.2]])
    
XWb =tf.matmul(X,W)+b

y=tf.nn.relu(tf.matmul(X,W)+b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb:')    
    print(sess.run(XWb ))    
    print('y:')    
    print(sess.run(y ))


# # y= sigmoid ( (X․W ) + b )

# In[3]:


X = tf.Variable([[0.4,0.2,0.4]])

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])

b = tf.Variable([[0.1,0.2]])

XWb=tf.matmul(X,W)+b

y=tf.nn.sigmoid(tf.matmul(X,W)+b)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb:')    
    print(sess.run(XWb))    
    print('y:')    
    print(sess.run(y ))


# # 以亂數產生Weight(W)與bais(b) 

# In[4]:


W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.Variable([[0.4,0.2,0.4]])
y=tf.nn.relu(tf.matmul(X,W)+b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('b:')
    print(sess.run(b ))    
    print('W:')
    print(sess.run(W ))
    print('y:')
    print(sess.run(y ))    


# In[5]:


W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.Variable([[0.4,0.2,0.4]])
y=tf.nn.relu(tf.matmul(X,W)+b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    (_b,_W,_y)=sess.run((b,W,y))
    print('b:')
    print(_b)
    print('W:')
    print(_W)
    print('y:')
    print(_y)   


# # placeholder

# In[6]:


W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.placeholder("float", [None,3])
y=tf.nn.relu(tf.matmul(X,W)+b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2,0.4]])
    (_b,_W,_X,_y)=sess.run((b,W,X,y),feed_dict={X:X_array})
    print('b:')
    print(_b)    
    print('W:')
    print(_W)
    print('X:')
    print(_X)
    print('y:')
    print(_y)


# In[7]:


_y.shape


# In[8]:


ts_norm = tf.random_normal([1000])
with tf.Session() as session:
    norm_data=ts_norm.eval()
print(len(norm_data))
print(norm_data[:30])


# In[9]:


import matplotlib.pyplot as plt
plt.hist(norm_data)
plt.show()    


# In[10]:


W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.placeholder("float", [None,3])
y=tf.nn.sigmoid(tf.matmul(X,W)+b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2 ,0.4],
                        [0.3,0.4 ,0.5],
                        [0.3,-0.4,0.5]])    
    (_b,_W,_X,_y)=sess.run((b,W,X,y),feed_dict={X:X_array})
    print('b:')
    print(_b)    
    print('W:')
    print(_W)
    print('X:')
    print(_X)
    print('y:')
    print(_y)


# In[11]:


def layer(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs


# In[12]:


X = tf.placeholder("float", [None,4])

y=layer(output_dim=3,input_dim=4,inputs=X,
        activation=tf.nn.relu)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2 ,0.4,0.1],
                        [0.3,0.4 ,0.5,0.3],
                        [0.3,-0.4,0.5,0.2]])    
    (_X,_y)=sess.run((X,y),feed_dict={X:X_array})
    print('X:')
    print(_X)
    print('y:')
    print(_y)    


# In[13]:


X = tf.placeholder("float", [None,4])
h=layer(output_dim=3,input_dim=4,inputs=X,
        activation=tf.nn.relu)
y=layer(output_dim=2,input_dim=3,inputs=h)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2 ,0.4,0.5]])    
    (layer_X,layer_h,layer_y)=             sess.run((X,h,y),feed_dict={X:X_array})
    print('input Layer X:')
    print(layer_X)
    print('hidden Layer h:')
    print(layer_h)
    print('output Layer y:')
    print(layer_y)


# In[14]:


def layer_debug(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs,W,b


# In[15]:


X = tf.placeholder("float", [None,4])
h,W1,b1=layer_debug(output_dim=3,input_dim=4,inputs=X,
                    activation=tf.nn.relu)
y,W2,b2=layer_debug(output_dim=2,input_dim=3,inputs=h)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2 ,0.4,0.5]])    
    (layer_X,layer_h,layer_y,W1,b1,W2,b2)=              sess.run((X,h,y,W1,b1,W2,b2),feed_dict={X:X_array})
    print('input Layer X:')
    print(layer_X)
    print('W1:')
    print(  W1)    
    print('b1:')
    print(  b1)    
    print('hidden Layer h:')
    print(layer_h)    
    print('W2:')
    print(  W2)    
    print('b2:')
    print(  b2)    
    print('output Layer y:')
    print(layer_y)

