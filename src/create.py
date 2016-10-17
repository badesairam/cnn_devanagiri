import tensorflow as tf 
from skimage import io
import numpy as np 
import os 
import scipy.misc
from skimage.morphology import square,dilation
from skimage.transform import rescale
from scipy.ndimage.filters import gaussian_filter

size = 64
def load_reshape(f):
	image = io.imread(f)
	image1 = np.invert(image)
	im1 = dilation(image1,square(7))
	# im2 = np.invert(im1)
	# im2 = gaussian_filter(image,sigma = 5)
	return rescale(im1,0.2)
def create_data(path_dir):
	train_img = io.ImageCollection(path_dir+'/*.png',conserve_memory=False,load_func=load_reshape)
	train_length = len(train_img)
	train_data = np.zeros([train_length,size*size],dtype=np.uint8)
	train_labels = np.zeros([train_length,104],dtype=np.uint8)
	#read labels
	labels_open = open(path_dir+"/labels.txt", "r").read().split('\n')
	labels_open.pop()
	labels = [int(x) for x in labels_open]
	for i in range(0,train_length):
		train_data[i] = train_img[i].reshape(1,size*size)
		#make a 104x1 label array
		label_vector = np.zeros([1,104],dtype=np.uint8)
		if (i%100==0):
			print (i)
		label_vector[0][labels[i]] = 1
		train_labels[i] = label_vector
	del train_img 
	del labels_open
	del labels
	return train_data,train_labels

###train data
print ("creating train_data")
train_data,train_labels = create_data('train')
print (train_data.shape)
print ("creating test_data")
test_data,test_labels = create_data('valid')