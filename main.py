import argparse
from net import *

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 8964

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--phase', type=str, default='Test',
					help='Train/Test network.')

class MODEL(Network):
	"""docstring for MODEL"""
	def __init__(self):
		Network.__init__(self)
		self.log_dir = 'pretrained'
		self.eval_file = './dataset/r3d_test.txt'		
		self.loss_type = 'balanced'

	def convert_one_hot_to_image(self, one_hot, dtype='float', act=None):
		if act == 'softmax':
			one_hot = tf.nn.softmax(one_hot, axis=-1)

		[n, h, w, c] = one_hot.shape.as_list()

		im = tf.reshape(tf.argmax(one_hot, axis=-1), [n, h, w, 1])
		if dtype == 'int':
			im = tf.cast(im, dtype=tf.uint8)
		else:
			im = tf.cast(im, dtype=tf.float32)
		return im

	def cross_two_tasks_weight(self, y1, y2):
		p1 = tf.reduce_sum(y1)
		p2 = tf.reduce_sum(y2)

		w1 = p2 / (p1 + p2)
		w2 = p1 / (p1 + p2)

		return w1, w2

	def balanced_entropy(self, x, y):
		# cliped_by_eps
		eps = 1e-6
		z = tf.nn.softmax(x)
		cliped_z = tf.clip_by_value(z, eps, 1-eps)
		log_z = tf.log(cliped_z)

		num_classes = y.shape.as_list()[-1]
		ind = tf.argmax(y, -1, output_type=tf.int32)
		# ind = tf.reshape(ind, shape=[1, 512, 512, 1]) # for debugging

		total = tf.reduce_sum(y) # total foreground pixels

		m_c = [] # index mask
		n_c = [] # each class foreground pixels
		for c in range(num_classes):
			m_c.append(tf.cast(tf.equal(ind, c), dtype=tf.int32))
			n_c.append(tf.cast(tf.reduce_sum(m_c[-1]), dtype=tf.float32))

		# compute count
		c = []
		for i in range(num_classes):
			c.append(total - n_c[i])
		tc = tf.add_n(c)

		# use for compute loss
		loss = 0.
		for i in range(num_classes): 
			w = c[i] / tc
			m_c_one_hot = tf.one_hot((i*m_c[i]), num_classes, axis=-1)
			y_c = m_c_one_hot*y

			loss += w*tf.reduce_mean(-tf.reduce_sum(y_c*log_z, axis=1)) 

		return (loss / num_classes) # mean 

	def train(self, loader_dict, num_batch, max_step=40000):
		images	 = loader_dict['images']
		labels_r_hot = loader_dict['label_rooms']
		labels_cw_hot  = loader_dict['label_boundaries']

		max_ep = max_step // num_batch
		print('max_step = {}, max_ep = {}, num_batch = {}'.format(max_step, max_ep, num_batch))

		logits1, logits2 = self.forward(images, init_with_pretrain_vgg=False)

		if self.loss_type == 'balanced':
			# in-task loss balance
			loss1 = self.balanced_entropy(logits1, labels_r_hot) # multi classes balance
			loss2 = self.balanced_entropy(logits2, labels_cw_hot)
		else:
			loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1, labels=labels_r_hot, name='bce1'))
			loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=labels_cw_hot, name='bce2'))

		# compute cross loss balance weight
		w1, w2 = self.cross_two_tasks_weight(labels_r_hot, labels_cw_hot)
		loss = (w1*loss1 + w2*loss2)

		optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, colocate_gradients_with_ops=True) # gradient ops assign to same device as forward ops

		# # add image summary
		# tf.summary.image('input', images)
		# tf.summary.image('label_r', self.convert_one_hot_to_image(labels_r_hot))
		# tf.summary.image('predict_room', self.convert_one_hot_to_image(logits1, act='softmax')) # room type to use argmax to visualize
		# tf.summary.image('predict_close_wall', tf.nn.sigmoid(logits2)) # boundaries type to use argmax to visualize

		# # add scalar summary
		# tf.summary.scalar('bce', loss)

		# define session
		config = tf.ConfigProto(allow_soft_placement=True) 
		config.gpu_options.allow_growth=True # prevent the program occupies all GPU memory
		with tf.Session(config=config) as sess:
			# init all variables in graph
			sess.run(tf.group(tf.global_variables_initializer(),
							tf.local_variables_initializer()))

			# saver 
			saver = tf.train.Saver(max_to_keep=10) 

			# filewriter for log info
			# log_dir = self.log_dir+'/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
			# writer = tf.summary.FileWriter(log_dir)
			# merged = tf.summary.merge_all()

			# coordinator for queue runner
			coord = tf.train.Coordinator()

			# start queue 
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			print("Start Training!")
			total_times = 0			

			for ep in range(max_ep): # epoch loop
				for n in range(num_batch): # batch loop
					tic = time.time()
					# [loss_value, update_value, summaries] = sess.run([loss, optim, merged])	
					[loss_value, update_value] = sess.run([loss, optim])	
					duration = time.time()-tic

					total_times += duration

					step = int(ep*num_batch + n)
					# write log 
					print('step {}: loss = {:.3}; {:.2} data/sec, excuted {} minutes'.format(step,
						loss_value, 1.0/duration, int(total_times/60)))
					# writer.add_summary(summaries, global_step=step)
				# save model parameters after 2 epoch training
				if ep % 2 == 0:
					saver.save(sess, self.log_dir+'/model', global_step=ep)
					self.evaluate(sess=sess, epoch=ep)
			saver.save(sess, self.log_dir+'/model', global_step=max_ep)
			self.evaluate(sess=sess, epoch=max_ep)

			# close session
			coord.request_stop()
			coord.join(threads)			
			sess.close()	

	def infer(self, save_dir='out', resize=True, merge=True):
		print("generating test set of {}.... will save to [./{}]".format(self.eval_file, save_dir))
		room_dir = os.path.join(save_dir, 'room')
		close_wall_dir = os.path.join(save_dir, 'boundary')

		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		if not os.path.exists(room_dir):
			os.mkdir(room_dir)
		if not os.path.exists(close_wall_dir):
			os.mkdir(close_wall_dir)

		x = tf.placeholder(shape=[1, 512, 512, 3], dtype=tf.float32)

		logits1, logits2 = self.forward(x, init_with_pretrain_vgg=False)
		rooms = self.convert_one_hot_to_image(logits1, act='softmax', dtype='int')
		close_walls = self.convert_one_hot_to_image(logits2, act='softmax', dtype='int')

		config = tf.ConfigProto(allow_soft_placement=True)
		sess = tf.Session(config=config)
		sess.run(tf.group(tf.global_variables_initializer(),
						tf.local_variables_initializer()))

		saver = tf.train.Saver() # restore all parameters
		saver.restore(sess, save_path = tf.train.latest_checkpoint(self.log_dir))

		# infer one by one
		paths = open(self.eval_file, 'r').read().splitlines()
		paths = [p.split('\t')[0] for p in paths]	
		for p in paths:
			im = imread(p, mode='RGB')  
			im_x = imresize(im, (512,512,3)) / 255. # resize and normalize
			im_x = np.reshape(im_x, (1,512,512,3))

			[out1, out2] = sess.run([rooms, close_walls], feed_dict={x: im_x})
			if resize:
				# out1 = imresize(np.squeeze(out1), (im.shape[0], im.shape[1])) # resize back 
				# out2 = imresize(np.squeeze(out2), (im.shape[0], im.shape[1])) # resize back 
				out1_rgb = ind2rgb(np.squeeze(out1))
				out1_rgb = imresize(out1_rgb, (im.shape[0], im.shape[1])) # resize back 
				out2_rgb = ind2rgb(np.squeeze(out2), color_map=floorplan_boundary_map)
				out2_rgb = imresize(out2_rgb, (im.shape[0], im.shape[1])) # resize back 
			else:
				out1_rgb = ind2rgb(np.squeeze(out1))
				out2_rgb = ind2rgb(np.squeeze(out2), color_map=floorplan_boundary_map)

			if merge:
				out1 = np.squeeze(out1)
				out2 = np.squeeze(out2)
				out1[out2==2] = 10
				out1[out2==1] = 9
				# out3_rgb = ind2rgb(out1, color_map=floorplan_fuse_map_figure) # use for present
				out3_rgb = ind2rgb(out1, color_map=floorplan_fuse_map) # use for present

			name = p.split('/')[-1]	
			save_path1 = os.path.join(room_dir, name.split('.jpg')[0]+'_rooms.png')		
			save_path2 = os.path.join(close_wall_dir, name.split('.jpg')[0]+'_bd_rm.png')		
			save_path3 = os.path.join(save_dir, name.split('.jpg')[0]+'_rooms.png')		

			imsave(save_path1, out1_rgb)
			imsave(save_path2, out2_rgb)
			if merge:
				imsave(save_path3, out3_rgb)
			# imsave(save_path4, out4)
			
			print('Saving prediction: {}'.format(name))	

	def evaluate(self, sess, epoch, num_of_classes=11):
		x = tf.placeholder(shape=[1, 512, 512, 3], dtype=tf.float32)
		logits1, logits2 = self.forward(x, init_with_pretrain_vgg=False)
		predict_bd = self.convert_one_hot_to_image(logits2, act='softmax', dtype='int')
		predict_room = self.convert_one_hot_to_image(logits1, act='softmax', dtype='int')

		paths = open(self.eval_file, 'r').read().splitlines()
		image_paths = [p.split('\t')[0] for p in paths] # image 
		gt2_paths = [p.split('\t')[2] for p in paths] # 2 denote doors (and windows)
		gt3_paths = [p.split('\t')[3] for p in paths] # 3 denote rooms
		gt4_paths = [p.split('\t')[-1] for p in paths] # last one denote close wall

		n = len(paths)

		hist = np.zeros((num_of_classes, num_of_classes))
		for i in range(n):
			im = imread(image_paths[i], mode='RGB')
			# for fuse label
			dd = imread(gt2_paths[i], mode='L')
			rr = imread(gt3_paths[i], mode='RGB')
			cw = imread(gt4_paths[i], mode='L')

			im = imresize(im, (512, 512, 3)) / 255. # normalize input image
			im = np.reshape(im, (1,512,512,3))
			# merge label
			rr = imresize(rr, (512, 512, 3))
			rr_ind = rgb2ind(rr)
			cw = imresize(cw, (512, 512)) / 255
			dd = imresize(dd, (512, 512)) / 255
			cw = (cw>0.5).astype(np.uint8)
			dd = (dd>0.5).astype(np.uint8)
			rr_ind[cw==1] = 10
			rr_ind[dd==1] = 9

			# merge prediciton
			rm_ind, bd_ind = sess.run([predict_room, predict_bd], feed_dict={x: im})
			rm_ind = np.squeeze(rm_ind)
			bd_ind = np.squeeze(bd_ind)
			rm_ind[bd_ind==2] = 10
			rm_ind[bd_ind==1] = 9

			hist += fast_hist(rm_ind.flatten(), rr_ind.flatten(), num_of_classes)

		overall_acc = np.diag(hist).sum() / hist.sum()
		mean_acc = np.diag(hist) / (hist.sum(1) + 1e-6)
		# iu = np.diag(hist) / (hist.sum(1) + 1e-6 + hist.sum(0) - np.diag(hist))			
		mean_acc9 = (np.nansum(mean_acc[:7])+mean_acc[-2]+mean_acc[-1]) / 9.

		file = open('EVAL_'+self.log_dir, 'a')
		print('Model at epoch {}: overall accuracy = {:.4}, mean_acc = {:.4}'.format(epoch, overall_acc, mean_acc9), file=file)
		for i in range(mean_acc.shape[0]):
			if i not in [7 ,8]: # ingore class 7 & 8 
				print('\t\tepoch {}: {}th label: accuracy = {:.4}'.format(epoch, i, mean_acc[i]), file=file)		
		file.close()

def main(args):
	tf.set_random_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	model = MODEL()

	if args.phase.lower() == 'train':
		loader_dict, num_batch = data_loader_bd_rm_from_tfrecord(batch_size=1)

		# START TRAINING
		tic = time.time()
		model.train(loader_dict, num_batch)
		toc = time.time()
		print('total training + evaluation time = {} minutes'.format((toc-tic)/60))
	elif args.phase.lower() == 'test':	
		model.infer()
	else:
		pass

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
