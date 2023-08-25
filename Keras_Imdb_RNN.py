#!/usr/bin/env python
# coding: utf-8

# # 1. Import Library

# # 資料準備

# In[1]:


from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# In[2]:


import re
re_tag = re.compile(r'<[^>]+>')

def rm_tags(text):
    return re_tag.sub('', text)


# In[3]:


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


# In[4]:


y_train,train_text=read_files("train")


# In[5]:


y_test,test_text=read_files("test")


# In[6]:


token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text)


# In[7]:


x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


# In[8]:


x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=380)


# # 建立模型

# In[9]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN


# In[10]:


model = Sequential()


# In[11]:


model.add(Embedding(output_dim=32,
                    input_dim=3800, 
                    input_length=380))
model.add(Dropout(0.35))


# In[12]:


model.add(SimpleRNN(units=16))


# In[13]:


model.add(Dense(units=256,activation='relu' ))
model.add(Dropout(0.35))


# In[14]:


model.add(Dense(units=1,activation='sigmoid' ))


# In[15]:


model.summary()


# # 訓練模型

# In[16]:


model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


# In[17]:


train_history =model.fit(x_train, y_train,batch_size=100, 
                         epochs=10,verbose=2,
                         validation_split=0.2)


# In[18]:


import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[19]:


show_train_history(train_history,'acc','val_acc')


# In[20]:


show_train_history(train_history,'loss','val_loss')


# # 評估模型準確率

# In[21]:


scores = model.evaluate(x_test, y_test, verbose=1)
scores[1]


# # 預測機率

# In[22]:


probility=model.predict(x_test)


# In[23]:


probility[:10]


# In[24]:


for p in probility[12500:12510]:
    print(p)


# # 預測結果

# In[25]:


predict=model.predict_classes(x_test)


# In[26]:


predict[:10]


# In[27]:


predict.shape


# In[28]:


predict_classes=predict.reshape(25000)
predict_classes


# # 查看預測結果

# In[29]:


SentimentDict={1:'正面的',0:'負面的'}
def display_test_Sentiment(i):
    print(test_text[i])
    print('label真實值:',SentimentDict[y_test[i]],
          '預測結果:',SentimentDict[predict_classes[i]])


# In[30]:


display_test_Sentiment(2)


# In[31]:


display_test_Sentiment(3)


# In[32]:


predict_classes[12500:12510]


# In[33]:


display_test_Sentiment(12502)


# In[34]:


#預測新的影評


# In[35]:


input_text='''
I can't vote because I have not watched this movie yet. I've been wanting to watch this movie since the time they announced making it which is about 2 years ago (!)
I was planning to go with the family to see the anticipated movie but my nieces had school exams at the opening time so we all decided to wait for the next weekend. I was utterly shocked to learn yesterday that they pulled the movie from the Kuwaiti theaters "temporarily" so that the outrageous censorship system can remove some unwanted scenes.
The controversial gay "moment" according to my online research is barely there, so I can't find any logical reason for all the fuss that's been going on. And it was bad enough when fanatics and haters tried (in vain) to kill the movie with low ratings and negative reviews even before it was in the cinemas and I'm pretty sure most of those trolls never got the chance to watch the movie at that time.
Based on the trailers, I think the movie is very promising and entertaining and you can't simply overlook the tremendous efforts made to bring this beloved tale to life. To knock down hundreds of people's obvious hard work with unprofessional critique and negative reviews just for the sake of hatred is unfathomable. I hope people won't judge a movie before having the experience of watching it in the first place.
Impatiently waiting for the Kuwaiti cinemas to bring back the movie... 
'''


# In[36]:


input_seq = token.texts_to_sequences([input_text])


# In[37]:


print(input_seq[0])


# In[38]:


len(input_seq[0] )


# In[39]:


pad_input_seq  = sequence.pad_sequences(input_seq , maxlen=380)


# In[40]:


len(pad_input_seq[0])


# In[41]:


predict_result=model.predict_classes(pad_input_seq)


# In[42]:


predict_result


# In[43]:


predict_result[0][0]


# In[44]:


SentimentDict[predict_result[0][0]]


# In[45]:


def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq  = sequence.pad_sequences(input_seq , maxlen=380)
    predict_result=model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])


# In[46]:


#http://www.imdb.com/title/tt2771200/
#http://www.imdb.com/title/tt2771200


# In[47]:


predict_review('''
As a fan of the original Disney film (Personally I feel it's their masterpiece) I was taken aback to the fact that a new version was in the making. Still excited I had high hopes for the film. Most of was shattered in the first 10 minutes. Campy acting with badly performed singing starts off a long journey holding hands with some of the worst CGI Hollywood have managed to but to screen in ages.
A film that is over 50% GCI, should focus on making that part believable, unfortunately for this film, it's far from that. It looks like the original film was ripped apart frame by frame and the beautiful hand-painted drawings have been replaced with digital caricatures. Besides CGI that is bad, it's mostly creepy. As the little teacup boy will give me nightmares for several nights to come. Emma Watson plays the same character as she always does, with very little acting effort and very little conviction as Belle. Although I can see why she was cast in the film based on merits, she is far from the right choice for the role. Dan Stevens does alright under as some motion captured dead-eyed Beast, but his performance feels flat as well. Luke Evans makes for a great pompous Gaston, but a character that has little depth doesn't really make for a great viewing experience. Josh Gad is a great comic relief just like the original movie's LeFou. Other than that, none of the cast stands out enough for me to remember them. Human or CHI creature. I was just bored through out the whole experience. And for a project costing $160 000 000, I can see why the PR department is pushing it so hard because they really need to get some cash back on this pile of wet stinky CGI-fur!
All and all, I might be bias from really loving Disney's first adaptation. That for me marks the high-point of all their work, perfectly combining the skills of their animators along with some CGI in a majestic blend. This film however is more like the bucket you wash off your paintbrush in, it has all the same colors, but muddled with water and to thin to make a captivating story from. The film is quite frankly not worth your time, you would be better off watching the original one more time. 
''')


# In[48]:


predict_review('''
I was really looking forward to this film. Not only has Disney recently made excellent live-action versions of their animated masterpieces (Jungle Book, Cinderella), but the cast alone (Emma Watson, Ian McKellen, Kevin Kline) already seemed to make this one a sure hit. Well, not so much as it turns out.
Some of the animation is fantastic, but because characters like Cogsworth (the clock), Lumière (the candelabra) and Chip (the little tea cup) now look "realistic", they lose a lot of their animated predecessors' charm and actually even look kind of creepy at times. And ironically - unlike in the animated original - in this new realistic version they only have very limited facial expressions (which is a creative decision I can't for the life of me understand).
Even when it works: there can be too much of a good thing. The film is overstuffed with lush production design and cgi (which is often weirdly artificial looking though) but sadly lacking in charm and genuine emotion. If this were a music album, I'd say it is "over-produced" and in need of more soul and swing. The great voice talent in some cases actually seems wasted, because it drowns in a sea of visual effects that numbs all senses. The most crucial thing that didn't work for me, though, is the Beast. He just never looks convincing. The eyes somehow don't look like real eyes and they're always slightly off.
On the positive side, I really liked Gaston, and the actor who played him, Luke Evans, actually gave the perhaps most energized performance of all. Kevin Kline as Belle's father has little to do but to look fatherly and old, but he makes the most of his part. Speaking of Belle, now that I've seen the film, I think her role was miscast. I think someone like Rachel McAdams would actually have been a more natural, lively and perhaps a bit more feisty Belle than Emma Watson.
If you love the original, you might want to give this one a pass, it's really not that good (although at least the songs were OK). Also, I'd think twice before bringing small children; without cute animated faces, all those "realistic" looking creatures and devices can be rather frightening for a child. ''')


# In[49]:


predict_review('''
The original Beauty and the Beast was my favorite cartoon as a kid but it did have major plot holes. Why had no one else ever seen the castle or knew where it was? Didn't anyone miss the people who were cursed? All of that gets an explanation when the enchantress places her curse in the beginning. Why did Belle and her Father move to a small town? Her mother died and the father thought it as best to leave. I love the new songs and added lyrics to the originals. I like the way the cgi beast looks (just the face is CGi). I think Emma Watson is a perfect Belle who is outspoken, fearless, and different. The set design is perfect for the era in France.
I know a lot of people disagree but I found this remake with all its changes to be more enchanting, beautiful, and complete than the original 1991 movie. To each his own but I think everyone should see it for themselves. 
''')


# In[50]:


predict_review('''
"Beauty and the Beast" was stunning and gorgeous. Beautifully and artfully performed. Dazzlingly colorful and charming. Fresh and lighthearted. Wonderfully and magically enthralling. I laughed, I cried, I floated along with the music, I bawled and bawled as the spellbinding elegance and splendor took me to another place and time. I was 5 years old again enjoying the Wonderful World of Magic that is Disney, as if for the first time. I will gladly spend my money over and over to see that magic unfold. I willingly and happily concede Emma Watson to be Belle, for this generation of children, instead of Hermione. She has earned her place as a Disney princess and I applaud everyone who had a part in this piece of magic. Truly, Disney knows how to make us all children again. 
''')


# # serialize model to JSON

# In[51]:


model_json = model.to_json()
with open("SaveModel/Imdb_RNN_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("SaveModel/Imdb_RNN_model.h5")
print("Saved model to disk")


# In[52]:


[1.02,2.03,-1.2,2,2],[1,2.1,3,2.2,3.4,3.6]

