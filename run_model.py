import tensorflow as tf 
from skimage import io
import numpy as np 
import os 
import scipy.misc
from skimage.morphology import square,dilation
from skimage.transform import rescale
from scipy.ndimage.filters import gaussian_filter
size = 64
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,dtype = tf.float32)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,dtype = tf.float32)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')

def load_reshape(f):
	image = io.imread(f)
	image1 = np.invert(image)
	im1 = dilation(image1,square(7))
	im1 = rescale(im1,0.2)
	# im2 = np.invert(im1)
	# im2 = gaussian_filter(image,sigma = 5)
	return im1.astype(np.float32)

def get_probability(paths):
	ret = np.zeros([len(paths),104])
	W_conv1 = weight_variable([5,5,1,6])
	b_conv1 = bias_variable([6])
	W_conv2 = weight_variable([5, 5, 6, 50])
	b_conv2 = bias_variable([50])
	m = 13
	n= 13
	l = 50
	n_hidden = 300
	W_fc1 = weight_variable([m*n*l,n_hidden])
	b_fc1 = bias_variable([n_hidden])
	W_fc3 = weight_variable([n_hidden, 104])
	b_fc3 = bias_variable([104])

	saver = tf.train.Saver()
	sess = tf.Session()
	saver.restore(sess,"model/model.ckpt")
	print ("model restored")
	index = 0
	img_data = np.zeros([len(paths),size*size])
	for path in paths :
		image = load_reshape(path)
		img_data[index]  = image.reshape(1,size*size)

	img_data = tf.reshape(img_data,[-1,64,64,1])
	img_data = tf.cast(img_data,tf.float32)
	h_conv1 = conv2d(img_data, W_conv1) + b_conv1
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	siz = h_pool2.get_shape()
	m = 13 #siz[1]
	n = 13 #siz[2]
	l = 50 #siz[3]
	# print ("size",siz)

	h_pool3_flat = tf.reshape(h_pool2, [-1, m*n*l])
	h_fc1 = tf.tanh(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)	

	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)	
	return y_conv

# l = ['valid/0.png','valid/1.png','valid/2.png','valid/3.png','valid/4.png','valid/5.png','valid/6.png','valid/7.png','valid/8.png','valid/9.png']
# ret = get_probability(l)

# accuracy = 0
# for x in range(1,10):
# 	if(tf.argmax(test_labels[x])==tf.argmax(ret[x])):
# 		accuracy = accuracy +1