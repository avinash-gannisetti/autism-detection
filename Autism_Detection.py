#!/usr/bin/env python
# coding: utf-8

# # Project: Detecting Autism with Deep Learning

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import os
import shutil
import glob
import re

import sklearn
from sklearn.model_selection import train_test_split

import tensorflow
import keras
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D


# # Importing Data

# In[23]:


data = 'AutismDataset'
autistic = 'AutismDataset/train/Autistic'
non_autistic= 'AutismDataset/train/Non_Autistic'

dirlist=[autistic, non_autistic]
classes=['yes', 'no']
filepaths=[]
labels=[]
for i,j in zip(dirlist, classes):
    filelist=os.listdir(i)
    for f in filelist:
        filepath=os.path.join (i,f)
        filepaths.append(filepath)
        labels.append(j)  
        
Files=pd.Series(filepaths, name='filepaths')
Label=pd.Series(labels, name='labels')
train_df=pd.concat([Files,Label], axis=1)
train_df['labels']=pd.get_dummies(train_df['labels'])

train_df.head()


# In[24]:


from sklearn.utils import shuffle
train_df = shuffle(train_df)
train_df.head()


# In[25]:


print('the shape of train data =', train_df.shape)


# In[26]:


data = 'AutismDataset'
autistic = 'AutismDataset/test/Autistic'
non_autistic= 'AutismDataset/test/Non_Autistic'

dirlist=[autistic, non_autistic]
classes=['yes', 'no']
filepaths=[]
labels=[]
for i,j in zip(dirlist, classes):
    filelist=os.listdir(i)
    for f in filelist:
        filepath=os.path.join (i,f)
        filepaths.append(filepath)
        labels.append(j)  
        
Files=pd.Series(filepaths, name='filepaths')
Label=pd.Series(labels, name='labels')
test_df=pd.concat([Files,Label], axis=1)
test_df['labels']=pd.get_dummies(test_df['labels'])

test_df.head()


# In[27]:


test_df = shuffle(test_df)
test_df.head()


# In[28]:


print('the shape of test data =', test_df.shape)


# # Pre-processing Data

# In[29]:


import cv2
plt.figure(figsize=(12,8))
for i in range(15):
    random = np.random.randint(1,len(train_df))
    plt.subplot(3,5,i+1)
    plt.imshow(cv2.imread(train_df.loc[random,"filepaths"]))
    plt.title(train_df.loc[random, "labels"]) 
    plt.xticks([])
    plt.yticks([])
plt.show()


# In[30]:


from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

train_img = []
train_labels = []

for idx, row in train_df.iterrows():
    path = row['filepaths']
    austic = row['labels']
    image = load_img(path,target_size=(224,224))
    image_array = img_to_array(image)   
    train_img.append(image_array)
    train_labels.append(austic)
    x_train = train_img
    y_train = np.array(train_labels)
    
test_img = []
test_labels = []

for idx, row in test_df.iterrows():
    path = row['filepaths']
    austic = row['labels']
    image = load_img(path,target_size=(224,224))
    image_array = img_to_array(image)   
    test_img.append(image_array)
    test_labels.append(austic)
    x_test = test_img
    y_test = np.array(test_labels)


# In[33]:


print('the shape of x train =', x_train.shape)
print('the shape of y train =', y_train.shape)
print('\n')
print('the shape of x test =', x_test.shape)
print('the shape of y test =', y_test.shape)


# In[32]:


x_train = x_train/ 255
x_test = x_test/ 255


# In[34]:


datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,
                             featurewise_std_normalization=False,samplewise_std_normalization=False,
                             zca_whitening=False,rotation_range = 40,zoom_range = 0.3,
                             width_shift_range=0.25,height_shift_range=0.25,
                             horizontal_flip = True,vertical_flip=False)
datagen.fit(x_train)


# # Modelling

# ## Model 1: 3 Layer CNN with MaxPooling

# In[35]:


input_shape = (224,224,3)


# In[49]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(),metrics=['acc'])


# In[50]:


model.summary()


# In[51]:


history = model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(x_test, y_test))


# In[52]:


acc = history.history['acc']
val_acc =history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


# In[53]:


plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()


# ## Model 2: 3 Layer CNN with Average Pooling

# In[54]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(),metrics=['acc'])


# In[55]:


model.summary()


# In[56]:


history = model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(x_test, y_test))


# In[57]:


acc = history.history['acc']
val_acc =history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


# In[58]:


plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()


# ## Model 3: 5 Layer CNN with MaxPooling

# In[60]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(),metrics=['acc'])


# In[61]:


model.summary()


# In[62]:


history = model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(x_test, y_test))


# In[63]:


acc = history.history['acc']
val_acc =history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


# In[64]:


plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()


# ## Model 4: 5 Layer CNN with Average Pooling

# In[65]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(),metrics=['acc'])


# In[66]:


model.summary()


# In[67]:


history = model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(x_test, y_test))


# In[68]:


acc = history.history['acc']
val_acc =history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


# In[69]:


plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()


# ## Model 5: 7 Layer CNN with MaxPooling

# In[72]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(),metrics=['acc'])


# In[73]:


model.summary()


# In[74]:


history = model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(x_test, y_test))


# In[75]:


acc = history.history['acc']
val_acc =history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


# In[76]:


plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()


# ## Model 6: 7 Layer CNN with Average Pooling

# In[77]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=tensorflow.keras.optimizers.Adam(),metrics=['acc'])


# In[78]:


model.summary()


# In[79]:


history = model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(x_test, y_test))


# In[80]:


acc = history.history['acc']
val_acc =history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


# In[81]:


plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()

