
# coding: utf-8

# ##Age and Gender Classification Using Convolutional Neural Networks - Demo
# 
# This code is released with the paper:
# 
# Gil Levi and Tal Hassner, "Age and Gender Classification Using Convolutional Neural Networks," IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
# 
# If you find the code useful, please add suitable reference to the paper in your work.

# In[1]:

import os
import numpy as np
import time
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


#caffe_root = './caffe/' 
import sys
#sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

# Loading the mean image
path = './models/'
mean_filename=path+'./mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]


# Loading the age network
pathage = './age_net_definitions/'
age_net_pretrained=path+'./age_net.caffemodel'
age_net_model_file=pathage+'./deploy.prototxt'
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


# Loading the gender network
pathgender = './gender_net_definitions/'
gender_net_pretrained=path+'./gender_net.caffemodel'
gender_net_model_file=pathgender+'./deploy.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


# Labels
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']


# ## Reading and plotting the input image
example_image = './example_image.jpg'
input_image = caffe.io.load_image(example_image)
#_ = plt.imshow(input_image)


# Age prediction
since = time.time()
prediction = age_net.predict([input_image]) 
print("Latency = {} seconds.".format( time.time() - since) )
print 'predicted age:', age_list[prediction[0].argmax()]


# Gender prediction
since = time.time()
prediction = gender_net.predict([input_image]) 
print("Latency = {} seconds.".format( time.time() - since) )
print 'predicted gender:', gender_list[prediction[0].argmax()]
