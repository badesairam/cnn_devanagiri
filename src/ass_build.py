import tensorflow as tf 
from skimage import io
import numpy as np 
import os 
from skimage.transform import rescale

size = 64

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')

saver = tf.train.Saver()

def build(learning_rate,batch_size,n_hidden):
	x = tf.placeholder(tf.float32, [None, size*size])
	y_ = tf.placeholder(tf.float32,[None,104])
	x_image = tf.reshape(x, [-1,64,64,1])
	
	#first convolute layer
	W_conv1 = weight_variable([5,5,1,6])
	b_conv1 = bias_variable([6])

	h_conv1 = conv2d(x_image, W_conv1) + b_conv1
	h_pool1 = max_pool_2x2(h_conv1)

	#second convolution layer

	W_conv2 = weight_variable([5, 5, 6, 50])
	b_conv2 = bias_variable([50])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# # third convolution layer
	# W_conv3 = weight_variable([2, 2, 16, 32])
	# b_conv3 = bias_variable([32])

	# h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	# h_pool3 = max_pool_2x2(h_conv3)

	#fully connected layer 1 
	siz = h_pool2.get_shape()
	m = 13 #siz[1]
	n = 13 #siz[2]
	l = 50 #siz[3]
	print ("size",siz)
	W_fc1 = weight_variable([m*n*l,n_hidden])
	b_fc1 = bias_variable([n_hidden])

	h_pool3_flat = tf.reshape(h_pool2, [-1, m*n*l])
	h_fc1 = tf.tanh(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)	

	# W_fc2 = weight_variable([n_hidden,n_hidden1])
	# b_fc2 = bias_variable([n_hidden1])

	# h_fc2 = tf.tanh(tf.matmul(h_fc1,W_fc2)+b_fc2)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# h_fc2 = tf.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	# h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

	# W_fc4 = weight_variable([n_hidden1,200])
	# # b_fc4 = bias_variable([200])
	# h_fc4 = tf.matmul(h_fc1_drop, W_fc4) + b_fc4
	# h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

	W_fc3 = weight_variable([n_hidden, 104])
	b_fc3 = bias_variable([104])
	y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

	#cross entropy with l2 loss
	# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_) + 0.01 * tf.nn.l2_loss(W_fc1) + 0.01 * tf.nn.l2_loss(W_fc3) + 0.01 * tf.nn.l2_loss(b_fc1) + 0.01 * tf.nn.l2_loss(b_fc1))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(prediction, "float"))

	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	saver = tf.train.Saver()

	print("training_started1")
	# print("Model saved in file: %s" % save_path)

	# saver.restore(sess,"model.ckpt")


	for j in range(17):
		print("iteration %d",j)
		tot_data = np.concatenate((train_data,train_labels),axis=1)
		np.random.shuffle(tot_data)
		train1 = tot_data[:,0:train_data.shape[1]]
		labels1 = tot_data[:,train_data.shape[1]:train_data.shape[1]+104]
		for i in range(1,int(17200/batch_size)):
			loss,acc,_=sess.run([cross_entropy,accuracy,train_step],feed_dict={x:train1[(i-1)*batch_size:i*batch_size-1],y_:labels1[(i-1)*batch_size:i*batch_size-1],keep_prob : 0.5})
			if(i*batch_size%1000 == 0):
				print("error %g,train accuracy %g"%(loss,acc))
		print("test accuracy %g"%sess.run(accuracy,feed_dict={x: test_data, y_: test_labels,keep_prob : 1.0}))
	save_path = saver.save(sess,"model/model.ckpt")
	print("Model saved in file: %s" % save_path)
	return sess.run(W_fc1)

