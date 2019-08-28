"""
Please prepare the raw image datas save to one folder, 
makesure the path is match to the train_file/test_file.
"""

from tf_record import *

train_file = '../dataset/r2v_train.txt'
test_file = '../dataset/r2v_test.txt'

# debug
if __name__ == '__main__':
	# write to TFRecord
	train_paths = open(train_file, 'r').read().splitlines()
	# test_paths = open(test_file, 'r').read().splitlines()

	# write_record(train_paths, name='../dataset/jp_train.tfrecords')
	# write_record(test_paths, name='../dataset/newyork_test.tfrecords')

	# write_seg_record(train_paths, name='../dataset/jp_seg_train.tfrecords')
	# write_seg_record(train_paths, name='../dataset/newyork_seg_train.tfrecords')

	write_bd_rm_record(train_paths, name='../dataset/jp_train.tfrecords')
	# write_bd_rm_record(train_paths, name='../dataset/all_train3.tfrecords')

	# read from TFRecord
	# loader_list = read_record('../dataset/jp_train.tfrecords')
	# loader_list = read_seg_record('../dataset/jp_seg_train.tfrecords')

	# loader_list = read_bd_rm_record('../dataset/newyork_bd_rm_train.tfrecords')
	# loader_list = read_bd_rm_record('../dataset/jp_bd_rm_train.tfrecords')

	# images = loader_list['images']
	# bd_ind = loader_list['label_boundaries']
	# rm_ind = loader_list['label_rooms']

	# with tf.Session() as sess:
	# 	# init all variables in graph
	# 	sess.run(tf.group(tf.global_variables_initializer(),
	# 					tf.local_variables_initializer()))
		
	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	# 	image, bd, rm = sess.run([images, bd_ind, rm_ind])

	# 	print 'sess run image shape = ',image.shape
	# 	print 'sess run wall shape = ', bd.shape
	# 	print 'sess run room shape =', rm.shape

	# 	bd = np.argmax(np.squeeze(bd), axis=-1)
	# 	rm = np.argmax(np.squeeze(rm), axis=-1)
	# 	plt.subplot(231)
	# 	plt.imshow(np.squeeze(image))
	# 	plt.subplot(233)
	# 	plt.imshow(bd)
	# 	plt.subplot(234)
	# 	plt.imshow(rm)
	# 	plt.show()

	# 	coord.request_stop()
	# 	coord.join(threads)
	# 	sess.close()