#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image, ImageOps
import numpy as np


# In[2]:


IMG_SIZE = 64
IMG_CHANNELS = 3
DIR = 'E:/Projects/Neural Net/BE Project/Image Colorization/Dataset/data'


# In[6]:


training_data_x = []
training_data_y = []
print('Reading')
for images in os.listdir(DIR):
    path = os.path.join(DIR,images)
    image = Image.open(path)
    bw_image = ImageOps.grayscale(image)
    bw_image = bw_image.resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
    image = image.resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
    image = np.array(image)
    bw_image = np.array(bw_image)
    if(len(image.shape)!=3):
        image = np.stack((image,)*3, axis=-1)
    if(image.shape[2]!=3):
        image = image[:,:,:3]
    training_data_x.append(bw_image)
    training_data_y.append(image)

print('Resizing')
training_data_x = np.reshape(
    training_data_x, (-1, IMG_SIZE, IMG_SIZE, 1))

training_data_y = np.reshape(
    training_data_y, (-1, IMG_SIZE, IMG_SIZE, 3))

training_data_x = training_data_x
training_data_y = training_data_y

print('Saving')
np.save('train_data.npy', training_data_x)
np.save('train_labels.npy', training_data_y)
