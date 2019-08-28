import numpy as np
import tensorflow as tf # using tf 1.10.1

from tensorflow.contrib.slim.nets import vgg

import os
import sys
import glob
import time
import random

from scipy import ndimage
from scipy.misc import imread, imresize, imsave

sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import fast_hist
from tf_record import read_record, read_bd_rm_record

GPU_ID = '0'

def data_loader_bd_rm_from_tfrecord(batch_size=1):
	paths = open('../dataset/r3d_train.txt', 'r').read().splitlines()

	loader_dict = read_bd_rm_record('../dataset/r3d.tfrecords', batch_size=batch_size, size=512)

	num_batch = len(paths) // batch_size

	return loader_dict, num_batch

class Network(object):
	"""docstring for Network"""
	def __init__(self, dtype=tf.float32):
		print 'Initial nn network object...'
		self.dtype = dtype
		self.pre_train_restore_map = {'vgg_16/conv1/conv1_1/weights':'FNet/conv1_1/W', # {'checkpoint_scope_var_name':'current_scope_var_name'} shape must be the same
									'vgg_16/conv1/conv1_1/biases':'FNet/conv1_1/b',	
									'vgg_16/conv1/conv1_2/weights':'FNet/conv1_2/W',
									'vgg_16/conv1/conv1_2/biases':'FNet/conv1_2/b',	
									'vgg_16/conv2/conv2_1/weights':'FNet/conv2_1/W',
									'vgg_16/conv2/conv2_1/biases':'FNet/conv2_1/b',	
									'vgg_16/conv2/conv2_2/weights':'FNet/conv2_2/W',
									'vgg_16/conv2/conv2_2/biases':'FNet/conv2_2/b',	
									'vgg_16/conv3/conv3_1/weights':'FNet/conv3_1/W',
									'vgg_16/conv3/conv3_1/biases':'FNet/conv3_1/b',	
									'vgg_16/conv3/conv3_2/weights':'FNet/conv3_2/W',
									'vgg_16/conv3/conv3_2/biases':'FNet/conv3_2/b',	
									'vgg_16/conv3/conv3_3/weights':'FNet/conv3_3/W',
									'vgg_16/conv3/conv3_3/biases':'FNet/conv3_3/b',	
									'vgg_16/conv4/conv4_1/weights':'FNet/conv4_1/W',
									'vgg_16/conv4/conv4_1/biases':'FNet/conv4_1/b',	
									'vgg_16/conv4/conv4_2/weights':'FNet/conv4_2/W',
									'vgg_16/conv4/conv4_2/biases':'FNet/conv4_2/b',	
									'vgg_16/conv4/conv4_3/weights':'FNet/conv4_3/W',
									'vgg_16/conv4/conv4_3/biases':'FNet/conv4_3/b',	
									'vgg_16/conv5/conv5_1/weights':'FNet/conv5_1/W',
									'vgg_16/conv5/conv5_1/biases':'FNet/conv5_1/b',	
									'vgg_16/conv5/conv5_2/weights':'FNet/conv5_2/W',
									'vgg_16/conv5/conv5_2/biases':'FNet/conv5_2/b',	
									'vgg_16/conv5/conv5_3/weights':'FNet/conv5_3/W',
									'vgg_16/conv5/conv5_3/biases':'FNet/conv5_3/b'} 

	# basic layer 
	def _he_uniform(self, shape, regularizer=None, trainable=None, name=None):
		name = 'W' if name is None else name+'/W'

		# size = (k_h, k_w, in_dim, out_dim)
		kernel_size = np.prod(shape[:2]) # k_h*k_w
		fan_in = shape[-2]*kernel_size  # fan_out = shape[-1]*kernel_size

		# compute the scale value
		s = np.sqrt(1. /fan_in)

		# create variable and specific GPU device
		with tf.device('/device:GPU:'+GPU_ID):
			w = tf.get_variable(name, shape, dtype=self.dtype,
							initializer=tf.random_uniform_initializer(minval=-s, maxval=s),
							regularizer=regularizer, trainable=trainable)

		return w

	def _constant(self, shape, value=0, regularizer=None, trainable=None, name=None):
		name = 'b' if name is None else name+'/b'

		with tf.device('/device:GPU:'+GPU_ID):
			b = tf.get_variable(name, shape, dtype=self.dtype,
							initializer=tf.constant_initializer(value=value),
							regularizer=regularizer, trainable=trainable)

		return b

	def _conv2d(self, tensor, dim, size=3, stride=1, rate=1, pad='SAME', act='relu', norm='none', G=16, bias=True, name='conv'):
		"""pre activate => norm => conv
		"""
		in_dim = tensor.shape.as_list()[-1]
		size = size if isinstance(size, (tuple, list)) else [size, size]
		stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
		rate = rate if isinstance(rate, (tuple, list)) else [1, rate, rate, 1]
		kernel_shape = [size[0], size[1], in_dim, dim]

		w = self._he_uniform(kernel_shape, name=name)
		b = self._constant(dim, name=name) if bias else 0

		if act == 'relu':
			tensor = tf.nn.relu(tensor, name=name+'/relu')
		elif act == 'sigmoid':
			tensor = tf.nn.sigmoid(tensor, name=name+'/sigmoid')
		elif act == 'softplus':
			tensor = tf.nn.softplus(tensor, name=name+'/softplus')
		elif act =='leaky_relu':
			tensor = tf.nn.leaky_relu(tensor, name=name+'/leaky_relu')
		else:
			norm = 'none'

		if norm == 'gn': # group normalization after acitvation
			# normalize
			# tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
			x = tf.transpose(tensor, [0, 3, 1, 2])
			N, C, H, W = x.get_shape().as_list()
			G = min(G, C)
			x = tf.reshape(x, [-1, G, C // G, H, W])
			mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
			x = (x - mean) / tf.sqrt(var + 1e-6)

			# per channel gamma and beta
			with tf.device('/device:GPU:'+GPU_ID):
				gamma = tf.get_variable(name+'/gamma', [C], dtype=self.dtype, initializer=tf.constant_initializer(1.0))
				beta = tf.get_variable(name+'/beta', [C], dtype=self.dtype, initializer=tf.constant_initializer(0.0))
				gamma = tf.reshape(gamma, [1, C, 1, 1])
				beta = tf.reshape(beta, [1, C, 1, 1])

			tensor = tf.reshape(x, [-1, C, H, W]) * gamma + beta
			# tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
			tensor = tf.transpose(tensor, [0, 2, 3, 1])			

		out = tf.nn.conv2d(tensor, w, strides=stride, padding=pad, dilations=rate, name=name) + b # default no bias

		return out	

	def _upconv2d(self, tensor, dim, size=4, stride=2, pad='SAME', act='relu', name='upconv'):
		[batch_size, h, w, in_dim] = tensor.shape.as_list()

		size = size if isinstance(size, (tuple, list)) else [size, size]
		stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]

		kernel_shape = [size[0], size[1], dim, in_dim]
		W = self._he_uniform(kernel_shape, name=name)

		if pad == 'SAME':
			out_shape = [batch_size, h*stride[1], w*stride[2], dim]
		else:
			out_shape = [batch_size, (h-1)*stride[1]+size[0],
									(w-1)*strdie[2]+size[1], dim]

		out = tf.nn.conv2d_transpose(tensor, W, output_shape=tf.stack(out_shape),
									strides=stride, padding=pad, name=name)

		# reset shape information
		out.set_shape(out_shape)

		if act == 'relu':
			out = tf.nn.relu(out, name=name+'/relu')
		elif act == 'sigmoid':
			out = tf.nn.sigmoid(out, name=name+'/sigmoid')
		else:
			pass

		return out		


	def _max_pool2d(self, tensor, size=2, stride=2, pad='VALID'):
		size = size if isinstance(size, (tuple, list)) else [1, size, size, 1]
		stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
		# 
		size = [1, size[0], size[1], 1] if len(size)==2 else size
		stride = [1, stride[0], stride[1], 1] if len(stride)==2 else stride

		out = tf.nn.max_pool(tensor, size, stride, pad)

		return out

	# following three function used for combining context features
	def _constant_kernel(self, shape, value=1.0, diag=False, flip=False, regularizer=None, trainable=None, name=None):
		name = 'fixed_w' if name is None else name+'/fixed_w'

		with tf.device('/device:GPU:'+GPU_ID):
			if not diag:
				k = tf.get_variable(name, shape, dtype=self.dtype,
						initializer=tf.constant_initializer(value=value),
						regularizer=regularizer, trainable=trainable)
			else:
				w = tf.eye(shape[0], num_columns=shape[1])
				if flip:
					w = tf.reshape(w, (shape[0], shape[1], 1))
					w = tf.image.flip_left_right(w)
				w = tf.reshape(w, shape)
				k = tf.get_variable(name, None, dtype=self.dtype, # constant initializer dont specific shape
						initializer=w,
						regularizer=regularizer, trainable=trainable)				

		return k		

	def _context_conv2d(self, tensor, dim=1, size=7, diag=False, flip=False, stride=1, name='cconv'):
		"""
		Implement using identity matrix, combine neighbour pixels without bias, current only accept depth 1 of input tensor

		Args:
			diag: create diagnoal identity matrix
			transpose: transpose the diagnoal matrix
		"""	
		in_dim = tensor.shape.as_list()[-1] # suppose to be 1
		size = size if isinstance(size, (tuple, list)) else [size, size]
		stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
		kernel_shape = [size[0], size[1], in_dim, dim]		

		w = self._constant_kernel(kernel_shape, diag=diag, flip=flip, trainable=False, name=name)
		out = tf.nn.conv2d(tensor, w, strides=stride, padding='SAME', name=name)

		return out

	def _non_local_context(self, tensor1, tensor2, stride=4, name='non_local_context'):
		"""Use 1/stride image size of identity one rank kernel to combine context features, default is half image size, embedding between encoder and decoder part
		
		Args:
			stride: define the neighbour size 
		"""
		assert tensor1.shape.as_list() == tensor2.shape.as_list(), "input tensor should have same shape"

		[N, H, W, C] = tensor1.shape.as_list()

		hs = H // stride if (H // stride) > 1 else (stride-1)
		vs = W // stride if (W // stride) > 1 else (stride-1)

		hs = hs if (hs%2!=0) else hs+1
		vs = hs if (vs%2!=0) else vs+1

		# compute attention map
		a = self._conv2d(tensor1, dim=C, name=name+'/fa1')
		a = self._conv2d(a, dim=C, name=name+'/fa2')
		a = self._conv2d(a, dim=1, size=1, act='linear', norm=None, name=name+'/a')
		a = tf.nn.sigmoid(a, name=name+'/a_sigmoid')

		# reduce the tensor depth 		
		x = self._conv2d(tensor2, dim=C, name=name+'/fx1')
		x = self._conv2d(x, dim=1, size=1, act='linear', norm=None, name=name+'/x')

		# pre attention, prevent the text
		x = a*x

		h = self._context_conv2d(x, size=[hs, 1], name=name+'/cc_h') # h
		v = self._context_conv2d(x, size=[1, vs], name=name+'/cc_v') # v
		d1 = self._context_conv2d(x, size=[hs, vs], diag=True, name=name+'/cc_d1') # d
		d2 = self._context_conv2d(x, size=[hs, vs], diag=True, flip=True, name=name+'/cc_d2') # d_t

		# double attention, prevent blurring
		c1 = a*(h+v+d1+d2)
		# c1 = (h+v+d1+d2)

		# expand to dim 
		c1 = self._conv2d(c1, dim=C, size=1, act='linear', norm=None, name=name+'/expand')
		# c1 = self._conv2d(c1, dim=C, name=name+'/conv1') # contextural feature

		# further convolution to learn richer feature
		features = tf.concat([tensor2, c1], axis=3, name=name+'/in_context_concat')
		out = self._conv2d(features, dim=C, name=name+'/conv2')

		# return out, a
		return out, None

	def _up_bilinear(self, tensor, dim, shape, name='upsample'):
		# [N, H, W, C] = tensor.shape.as_list()

		out = self._conv2d(tensor, dim=dim, size=1, act='linear', name=name+'/1x1_conv')
		return tf.image.resize_images(out, shape)


	def forward(self, inputs, init_with_pretrain_vgg=False, pre_trained_model='./vgg16/vgg_16.ckpt'):
		# feature extraction part and also the share network 
		reuse_fnet = len([v for v in tf.global_variables() if v.name.startswith('FNet')]) > 0
		with tf.variable_scope('FNet', reuse=reuse_fnet):
			# feature extraction
			self.conv1_1 = self._conv2d(inputs, dim=64, name='conv1_1') 
			self.conv1_2 = self._conv2d(self.conv1_1, dim=64, name='conv1_2')		
			self.pool1  = self._max_pool2d(self.conv1_2) # 256 => /2			

			self.conv2_1 = self._conv2d(self.pool1, dim=128, name='conv2_1')
			self.conv2_2 = self._conv2d(self.conv2_1, dim=128, name='conv2_2')	
			self.pool2 = self._max_pool2d(self.conv2_2) # 128 => /4		

			self.conv3_1 = self._conv2d(self.pool2, dim=256, name='conv3_1')
			self.conv3_2 = self._conv2d(self.conv3_1, dim=256, name='conv3_2')	
			self.conv3_3 = self._conv2d(self.conv3_2, dim=256, name='conv3_3')	
			self.pool3 = self._max_pool2d(self.conv3_3) # 64 => /8		

			self.conv4_1 = self._conv2d(self.pool3, dim=512, name='conv4_1')	
			self.conv4_2 = self._conv2d(self.conv4_1, dim=512, name='conv4_2')		
			self.conv4_3 = self._conv2d(self.conv4_2, dim=512, name='conv4_3')	
			self.pool4 = self._max_pool2d(self.conv4_3)	# 32 => /16		

			self.conv5_1 = self._conv2d(self.pool4, dim=512, name='conv5_1')	
			self.conv5_2 = self._conv2d(self.conv5_1, dim=512, name='conv5_2')		
			self.conv5_3 = self._conv2d(self.conv5_2, dim=512, name='conv5_3')		
			self.pool5 = self._max_pool2d(self.conv5_3)	# 16 => /32		

			# init feature extraction part from pre-train vgg16
			if init_with_pretrain_vgg:
				tf.train.init_from_checkpoint(pre_trained_model, self.pre_train_restore_map)

			# input size for logits predict
			[n, h, w, c] = inputs.shape.as_list()

		reuse_cw_net = len([v for v in tf.global_variables() if v.name.startswith('CWNet')]) > 0
		with tf.variable_scope('CWNet', reuse=reuse_cw_net):
			# upsample
			up2 = (self._upconv2d(self.pool5, dim=256, act='linear', name='up2_1') # 32 => /16
					+ self._conv2d(self.pool4, dim=256, act='linear', name='pool4_s'))
			self.up2_cw = self._conv2d(up2, dim=256, name='up2_3')

			up4 = (self._upconv2d(self.up2_cw, dim=128, act='linear', name='up4_1') # 64 => /8
					+ self._conv2d(self.pool3, dim=128, act='linear', name='pool3_s'))
			self.up4_cw = self._conv2d(up4, dim=128, name='up4_3')

			up8 = (self._upconv2d(self.up4_cw, dim=64, act='linear', name='up8_1') # 128 => /4
					+ self._conv2d(self.pool2, dim=64, act='linear', name='pool2_s'))
			self.up8_cw = self._conv2d(up8, dim=64, name='up8_2')

			up16 = (self._upconv2d(self.up8_cw, dim=32, act='linear', name='up16_1') # 256 => /2
					+ self._conv2d(self.pool1, dim=32, act='linear', name='pool1_s'))
			self.up16_cw = self._conv2d(up16, dim=32, name='up16_2')

			# predict logits
			logits_cw = self._up_bilinear(self.up16_cw, dim=3, shape=(h, w), name='logits')	

		# decode network for room type detection
		reuse_rnet = len([v for v in tf.global_variables() if v.name.startswith('RNet')]) > 0
		with tf.variable_scope('RNet', reuse=reuse_rnet):
			# upsample
			up2 = (self._upconv2d(self.pool5, dim=256, act='linear', name='up2_1') # 32 => /16
					+ self._conv2d(self.pool4, dim=256, act='linear', name='pool4_s'))
			up2 = self._conv2d(up2, dim=256, name='up2_2')
			up2, _ = self._non_local_context(self.up2_cw, up2, name='context_up2')

			up4 = (self._upconv2d(up2, dim=128, act='linear', name='up4_1') # 64 => /8
					+ self._conv2d(self.pool3, dim=128, act='linear', name='pool3_s'))
			up4 = self._conv2d(up4, dim=128, name='up4_2')
			up4, _ = self._non_local_context(self.up4_cw, up4, name='context_up4')

			up8 = (self._upconv2d(up4, dim=64, act='linear', name='up8_1') # 128 => /4
					+ self._conv2d(self.pool2, dim=64, act='linear', name='pool2_s'))
			up8 = self._conv2d(up8, dim=64, name='up8_2')
			up8, _ = self._non_local_context(self.up8_cw, up8, name='context_up8')

			up16 = (self._upconv2d(up8, dim=32, act='linear', name='up16_1') # 256 => /2
					+ self._conv2d(self.pool1, dim=32, act='linear', name='pool1_s'))
			up16 = self._conv2d(up16, dim=32, name='up16_2')
			self.up16_r, self.a = self._non_local_context(self.up16_cw, up16, name='context_up16')

			# predict logits
			logits_r = self._up_bilinear(self.up16_r, dim=9, shape=(h, w), name='logits')	

			return logits_r, logits_cw	