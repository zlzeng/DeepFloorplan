import numpy as np

import tensorflow as tf

from scipy.misc import imread, imresize, imsave
from matplotlib import pyplot as plt
from rgb_ind_convertor import *

import os
import sys
import glob 
import time

def load_raw_images(path):
	paths = path.split('\t')

	image = imread(paths[0], mode='RGB')
	wall  = imread(paths[1], mode='L')
	close = imread(paths[2], mode='L')
	room  = imread(paths[3], mode='RGB')
	close_wall = imread(paths[4], mode='L')

	# NOTE: imresize will rescale the image to range [0, 255], also cast data into uint8 or uint32
	image = imresize(image, (512, 512, 3))
	wall = imresize(wall, (512, 512))
	close = imresize(close, (512, 512))
	close_wall = imresize(close_wall, (512, 512))
	room = imresize(room, (512, 512, 3))

	room_ind = rgb2ind(room)

	# make sure the dtype is uint8
	image = image.astype(np.uint8)
	wall = wall.astype(np.uint8)
	close = close.astype(np.uint8)
	close_wall = close_wall.astype(np.uint8)
	room_ind = room_ind.astype(np.uint8)

	# debug
	# plt.subplot(231)
	# plt.imshow(image)
	# plt.subplot(233)
	# plt.imshow(wall, cmap='gray')
	# plt.subplot(234)
	# plt.imshow(close, cmap='gray')
	# plt.subplot(235)
	# plt.imshow(room_ind)
	# plt.subplot(236)
	# plt.imshow(close_wall, cmap='gray')
	# plt.show()

	return image, wall, close, room_ind, close_wall

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_record(paths, name='dataset.tfrecords'):
	writer = tf.python_io.TFRecordWriter(name)
	
	for i in range(len(paths)):
		# Load the image
		image, wall, close, room_ind, close_wall = load_raw_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
					'wall': _bytes_feature(tf.compat.as_bytes(wall.tostring())),
					'close': _bytes_feature(tf.compat.as_bytes(close.tostring())),
					'room': _bytes_feature(tf.compat.as_bytes(room_ind.tostring())),
					'close_wall': _bytes_feature(tf.compat.as_bytes(close_wall.tostring()))}
		
		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))
    
		# Serialize to string and write on the file
		writer.write(example.SerializeToString())
    
	writer.close()

def read_record(data_path, batch_size=1, size=512):
	feature = {'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'wall': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'close': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'room': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'close_wall': tf.FixedLenFeature(shape=(), dtype=tf.string)}

	# Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)
	
	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)

	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['image'], tf.uint8)
	wall = tf.decode_raw(features['wall'], tf.uint8)
	close = tf.decode_raw(features['close'], tf.uint8)
	room = tf.decode_raw(features['room'], tf.uint8)
	close_wall = tf.decode_raw(features['close_wall'], tf.uint8)

	# Cast data
	image = tf.cast(image, dtype=tf.float32)
	wall = tf.cast(wall, dtype=tf.float32)
	close = tf.cast(close, dtype=tf.float32)
	# room = tf.cast(room, dtype=tf.float32)
	close_wall = tf.cast(close_wall, dtype=tf.float32)

	# Reshape image data into the original shape
	image = tf.reshape(image, [size, size, 3])
	wall = tf.reshape(wall, [size, size, 1])
	close = tf.reshape(close, [size, size, 1])
	room = tf.reshape(room, [size, size])
	close_wall = tf.reshape(close_wall, [size, size, 1])


	# Any preprocessing here ...
	# normalize 
	image = tf.divide(image, tf.constant(255.0))
	wall = tf.divide(wall, tf.constant(255.0))
	close = tf.divide(close, tf.constant(255.0))
	close_wall = tf.divide(close_wall, tf.constant(255.0))

	# Genereate one hot room label
	room_one_hot = tf.one_hot(room, 9, axis=-1)

	# Creates batches by randomly shuffling tensors
	images, walls, closes, rooms, close_walls = tf.train.shuffle_batch([image, wall, close, room_one_hot, close_wall], 
						batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)	

	# images, walls = tf.train.shuffle_batch([image, wall], 
						# batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)	

	return {'images': images, 'walls': walls, 'closes': closes, 'rooms': rooms, 'close_walls': close_walls}
	# return {'images': images, 'walls': walls}

# ------------------------------------------------------------------------------------------------------------------------------------- *
# Following are only for segmentation task, merge all label into one 

def load_seg_raw_images(path):
	paths = path.split('\t')

	image = imread(paths[0], mode='RGB')
	close = imread(paths[2], mode='L')
	room  = imread(paths[3], mode='RGB')
	close_wall = imread(paths[4], mode='L')

	# NOTE: imresize will rescale the image to range [0, 255], also cast data into uint8 or uint32
	image = imresize(image, (512, 512, 3))
	close = imresize(close, (512, 512)) / 255
	close_wall = imresize(close_wall, (512, 512)) / 255
	room = imresize(room, (512, 512, 3))

	room_ind = rgb2ind(room)

	# merge result
	d_ind = (close>0.5).astype(np.uint8)
	cw_ind = (close_wall>0.5).astype(np.uint8)
	room_ind[cw_ind==1] = 10
	room_ind[d_ind==1] = 9

	# make sure the dtype is uint8
	image = image.astype(np.uint8)
	room_ind = room_ind.astype(np.uint8)

	# debug
	# merge = ind2rgb(room_ind, color_map=floorplan_fuse_map)
	# plt.subplot(131)
	# plt.imshow(image)
	# plt.subplot(132)
	# plt.imshow(room_ind)
	# plt.subplot(133)
	# plt.imshow(merge/256.)
	# plt.show()

	return image, room_ind

def write_seg_record(paths, name='dataset.tfrecords'):
	writer = tf.python_io.TFRecordWriter(name)
	
	for i in range(len(paths)):
		# Load the image
		image, room_ind = load_seg_raw_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
					'label': _bytes_feature(tf.compat.as_bytes(room_ind.tostring()))}
		
		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))
    
		# Serialize to string and write on the file
		writer.write(example.SerializeToString())
    
	writer.close()

def read_seg_record(data_path, batch_size=1, size=512):
	feature = {'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'label': tf.FixedLenFeature(shape=(), dtype=tf.string)}

	# Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)
	
	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)

	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['image'], tf.uint8)
	label = tf.decode_raw(features['label'], tf.uint8)

	# Cast data
	image = tf.cast(image, dtype=tf.float32)

	# Reshape image data into the original shape
	image = tf.reshape(image, [size, size, 3])
	label = tf.reshape(label, [size, size])


	# Any preprocessing here ...
	# normalize 
	image = tf.divide(image, tf.constant(255.0))

	# Genereate one hot room label
	label_one_hot = tf.one_hot(label, 11, axis=-1)

	# Creates batches by randomly shuffling tensors
	images, labels = tf.train.shuffle_batch([image, label_one_hot], 
						batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)	

	# images, walls = tf.train.shuffle_batch([image, wall], 
						# batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)	

	return {'images': images, 'labels': labels}

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- *
# ------------------------------------------------------------------------------------------------------------------------------------- *
# Following are only for multi-task network. Two labels(boundary and room.)

def load_bd_rm_images(path):
	paths = path.split('\t')

	image = imread(paths[0], mode='RGB')
	close = imread(paths[2], mode='L')
	room  = imread(paths[3], mode='RGB')
	close_wall = imread(paths[4], mode='L')

	# NOTE: imresize will rescale the image to range [0, 255], also cast data into uint8 or uint32
	image = imresize(image, (512, 512, 3))
	close = imresize(close, (512, 512)) / 255.
	close_wall = imresize(close_wall, (512, 512)) / 255.
	room = imresize(room, (512, 512, 3))

	room_ind = rgb2ind(room)

	# merge result
	d_ind = (close>0.5).astype(np.uint8)
	cw_ind = (close_wall>0.5).astype(np.uint8)

	cw_ind[cw_ind==1] = 2
	cw_ind[d_ind==1] = 1

	# make sure the dtype is uint8
	image = image.astype(np.uint8)
	room_ind = room_ind.astype(np.uint8)
	cw_ind = cw_ind.astype(np.uint8)

	# debugging
	# merge = ind2rgb(room_ind, color_map=floorplan_fuse_map)
	# rm = ind2rgb(room_ind)
	# bd = ind2rgb(cw_ind, color_map=floorplan_boundary_map)
	# plt.subplot(131)
	# plt.imshow(image)
	# plt.subplot(132)
	# plt.imshow(rm/256.)
	# plt.subplot(133)
	# plt.imshow(bd/256.)
	# plt.show()

	return image, cw_ind, room_ind, d_ind

def write_bd_rm_record(paths, name='dataset.tfrecords'):
	writer = tf.python_io.TFRecordWriter(name)
	
	for i in range(len(paths)):
		# Load the image
		image, cw_ind, room_ind, d_ind = load_bd_rm_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
					'boundary': _bytes_feature(tf.compat.as_bytes(cw_ind.tostring())),
					'room': _bytes_feature(tf.compat.as_bytes(room_ind.tostring())),
					'door': _bytes_feature(tf.compat.as_bytes(d_ind.tostring()))}
		
		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))
    
		# Serialize to string and write on the file
		writer.write(example.SerializeToString())
    
	writer.close()

def read_bd_rm_record(data_path, batch_size=1, size=512):
	feature = {'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'boundary': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'room': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'door': tf.FixedLenFeature(shape=(), dtype=tf.string)}

	# Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)
	
	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)

	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['image'], tf.uint8)
	boundary = tf.decode_raw(features['boundary'], tf.uint8)
	room = tf.decode_raw(features['room'], tf.uint8)
	door = tf.decode_raw(features['door'], tf.uint8)

	# Cast data
	image = tf.cast(image, dtype=tf.float32)

	# Reshape image data into the original shape
	image = tf.reshape(image, [size, size, 3])
	boundary = tf.reshape(boundary, [size, size])
	room = tf.reshape(room, [size, size])
	door = tf.reshape(door, [size, size])

	# Any preprocessing here ...
	# normalize 
	image = tf.divide(image, tf.constant(255.0))

	# Genereate one hot room label
	label_boundary = tf.one_hot(boundary, 3, axis=-1)
	label_room = tf.one_hot(room, 9, axis=-1)

	# Creates batches by randomly shuffling tensors
	images, label_boundaries, label_rooms, label_doors = tf.train.shuffle_batch([image, label_boundary, label_room, door], 
						batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)	

	# images, walls = tf.train.shuffle_batch([image, wall], 
						# batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)	

	return {'images': images, 'label_boundaries': label_boundaries, 'label_rooms': label_rooms, 'label_doors': label_doors}
