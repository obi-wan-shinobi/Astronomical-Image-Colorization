import os
from PIL import Image
import numpy as np


# In[2]:


IMG_SIZE = 64
IMG_CHANNELS = 3
DIR = 'E:/Projects/Neural Net/BE Project/Image Colorization/Dataset/data'


# In[6]:


training_data = []
print('Reading')
for images in os.listdir(DIR):
    path = os.path.join(DIR,images)
    image = Image.open(path).resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
    image = np.array(image)
    if(len(image.shape)!=3):
        print(image.shape)
        image = np.stack((image,)*3, axis=-1)
        print(image.shape)
    training_data.append(image)


print('Resizing')
print(len(training_data))

for i in range(len(training_data)):
    if(training_data[i].shape[2]!=3):
        training_data[i] = training_data[i][:,:,:3]

training_data = np.reshape(
    training_data, (-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

print(training_data.shape)

training_data = training_data / 127.5 - 1

print('Saving')
np.save('train_labels.npy', training_data)
