#!/usr/bin/env python
# coding: utf-8

# In[ ]:


x_train=train_datagon.flow_from_directory('/content/drive/MyDrive/dataset/train_set',target_size=(64,64),batch_size=5,color_mode='rgb',class_mode='categorical')
x_test=test_datagon.flow_from_directory('/content/drive/MyDrive/dataset/test_set',target_size=(64,64),batch_size=5,color_mode='rgb',class_mode='categorical')
Found 750 images belonging to 4 classes.
Found 198 images belonging to 4 classes.

