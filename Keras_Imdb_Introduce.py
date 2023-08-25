#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import os
import tarfile


# In[2]:


url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)


# In[3]:


if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('data/')


# # 1. Import Library

# In[4]:


from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# # 資料準備

# In[5]:





# In[6]:


#讀取檔案


# In[7]:


import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


# In[8]:


import os
def read_files(filetype):
    path = "data/aclImdb/"
    file_list=[]

    positive_path=path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    
    negative_path=path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype, 'files:',len(file_list))
       
    all_labels = ([1] * 12500 + [0] * 12500) 
    
    all_texts  = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
    return all_labels,all_texts


# In[9]:


y_train,train_text=read_files("train")


# In[10]:


y_test,test_text=read_files("test")


# In[11]:


#查看正面評價的影評


# In[12]:


train_text[0]


# In[13]:


y_train[0]


# In[14]:


#查看負面評價的影評


# In[15]:


train_text[12499]


# In[16]:


y_train[12499]


# # 先讀取所有文章建立字典，限制字典的數量為nb_words=2000

# In[17]:


token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)


# In[18]:


#Tokenizer屬性


# In[19]:


#fit_on_texts 讀取多少文章


# In[20]:


print(token.document_count)


# In[21]:


print(token.word_index)


# # 將每一篇文章的文字轉換一連串的數字
# #只有在字典中的文字會轉換為數字

# In[22]:


x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


# In[23]:


print(train_text[0])


# In[24]:


print(x_train_seq[0])


# # 讓轉換後的數字長度相同

# In[25]:


#
#文章內的文字，轉換為數字後，每一篇的文章地所產生的數字長度都不同，因為後需要進行類神經網路的訓練，所以每一篇文章所產生的數字長度必須相同
#以下列程式碼為例maxlen=100，所以每一篇文章轉換為數字都必須為100
#bj6eji3t03g/ 2k


# In[26]:


x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=100)


# In[27]:


#如果文章轉成數字大於0,pad_sequences處理後，會truncate前面的數字


# In[28]:


print('before pad_sequences length=',len(x_train_seq[0]))
print(x_train_seq[0])


# In[29]:


print('after pad_sequences length=',len(x_train[0]))
print(x_train[0])


# In[30]:


#如果文章轉成數字不足100,pad_sequences處理後，前面會加上0


# In[31]:


print('before pad_sequences length=',len(x_train_seq[1]))
print(x_train_seq[1])


# In[32]:


print('after pad_sequences length=',len(x_train[1]))
print(x_train[1])


# # 資料預處理

# In[33]:


token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)


# In[34]:


x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


# In[35]:


x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=100)

