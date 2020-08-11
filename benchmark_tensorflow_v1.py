#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import sys
import tqdm
import time
import argparse
import pathlib
import random
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2

import matplotlib.pyplot as plt

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='TensorFlow Low Level APIのサンプル', formatter_class=argparse.RawTextHelpFormatter)
	
	parser.add_argument('--train', dest='flg_train', action='store_true', help='セット時，モデルを生成')
	parser.add_argument('--model', dest='model', type=str, default=None, required=False, help='TensorFlow学習済みモデルを指定')
	
	return parser.parse_args()
	
#---------------------------------
# クラス
#---------------------------------
class TF_Model():
	def __init__(self):
		tf.compat.v1.disable_eager_execution()
		return
	
	def fc_net(self, input_dims=784, hidden1_dims=300, hidden2_dims=100, output_dims=10):
		x = tf.compat.v1.placeholder(tf.float32, [None, input_dims])
		
		W1 = tf.Variable(tf.truncated_normal([input_dims, hidden1_dims]))
		b1 = tf.Variable(tf.zeros([hidden1_dims]))
		h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
		
		W2 = tf.Variable(tf.truncated_normal([hidden1_dims, hidden2_dims]))
		b2 = tf.Variable(tf.zeros([hidden2_dims]))
		h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)
		
		W3 = tf.Variable(tf.truncated_normal([hidden2_dims, output_dims]))
		b3 = tf.Variable(tf.zeros([output_dims]))
		y = tf.add(tf.matmul(h2, W3), b3, name='output')
		
		y_ = tf.compat.v1.placeholder(tf.float32, [None, output_dims])
		
		return x, y, y_


	def conv_net(self, input_dims, conv_channels, conv_kernel_size, pool_size, fc_channels, output_dims, dropout_rate, 
			is_train=True):
		"""
			input_dims: 入力次元 [N, H, W, C]
			conv_channels: 畳み込み層のChannel数 [<layer1 channel>, <layer2 channel>, ...]
			conv_kernel_size: 畳み込み層のカーネルサイズ [<layer1 kernel size>, <layer2 kernel size>, ...]
			pool_size: プーリングサイズ [<layer1 pool size>, <layer2 kernel size>, ...]
			fc_channels: 全結合層のChannel数 [<layer1 channel>, <layer2 channel>, ...]
			output_dims: 出力次元
		"""
		
		def weight_variable(shape, scope, id=0):
			init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
			with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
				var = tf.compat.v1.get_variable('Weight{}'.format(id), shape=shape, initializer=init)
		
		def bias_variable(shape, scope, id=0):
			init = tf.constant_initializer([0.0])
			with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
				var = tf.compat.v1.get_variable('Bias{}'.format(id), shape=shape, initializer=init)
		
		def bn_variables(shape, scope):
			with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
				gamma = tf.compat.v1.get_variable('gamma', shape[-1], initializer=tf.constant_initializer(1.0))
				beta = tf.compat.v1.get_variable('beta', shape[-1], initializer=tf.constant_initializer(0.0))
				moving_avg = tf.compat.v1.get_variable('moving_avg', shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
				moving_var = tf.compat.v1.get_variable('moving_var', shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
		
		def batch_norm(x, scope, train, epsilon=0.001, decay=0.99):
			# --- Activationの後にBatchNorm層を入れる ---
			# Perform a batch normalization after a conv layer or a fc layer
			# gamma: a scale factor
			# beta: an offset
			# epsilon: the variance epsilon - a small float number to avoid dividing by 0
			
			if train:
				with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
					shape = x.get_shape().as_list()
					ema = tf.compat.v1.train.ExponentialMovingAverage(decay=decay)
					batch_avg, batch_var = tf.nn.moments(x, list(range(len(shape)-1)))
					
					print(batch_avg.name)
					print(batch_var.name)
					print(ema.name)
					print(x.name)
					
					ema_apply_op = ema.apply([batch_avg, batch_var])
					
			with tf.compat.v1.variable_scope(scope, reuse=True):
				gamma, beta = tf.compat.v1.get_variable('gamma'), tf.compat.v1.get_variable('beta')
				moving_avg, moving_var = tf.compat.v1.get_variable('moving_avg'), tf.compat.v1.get_variable('moving_var')
				control_inputs = []
				if train:
					with tf.control_dependencies([ema_apply_op]):
						avg = moving_avg.assign(ema.average(batch_avg))
						var = moving_var.assign(ema.average(batch_var))
						
						with tf.control_dependencies([avg, var]):
							control_inputs = [moving_avg, moving_var]
				else:
					avg = moving_avg
					var = moving_var
				with tf.control_dependencies(control_inputs):
					output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
			
			return output
		
		def conv2d(x, scope, id):
			with tf.compat.v1.variable_scope(scope, reuse=True):
				W = tf.compat.v1.get_variable('Weight{}'.format(id))
				b = tf.compat.v1.get_variable('Bias{}'.format(id))
				return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
		
		def affine(x, scope, id=0):
			with tf.compat.v1.variable_scope(scope, reuse=True):
				W = tf.compat.v1.get_variable('Weight{}'.format(id))
				b = tf.compat.v1.get_variable('Bias{}'.format(id))
				return tf.matmul(x, W) + b
		
		def max_pool(x, size):
			return tf.nn.max_pool2d(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
		
		x = tf.compat.v1.placeholder(tf.float32, input_dims)
		y_ = tf.compat.v1.placeholder(tf.float32, output_dims)
		
		# convolution layer
		h_out = x
		h_out_shape = input_dims[1:]
		prev_channel = input_dims[-1]
		for i, (_conv_channel, _conv_kernel_size, _pool_size) in enumerate(zip(conv_channels, conv_kernel_size, pool_size)):
			print(_conv_channel, _conv_kernel_size, _pool_size, prev_channel)
			
			for ii in range(2):
				weight_variable([_conv_kernel_size, _conv_kernel_size, prev_channel, _conv_channel], 'ConvLayer{}'.format(i), ii)
				bias_variable([_conv_channel], 'ConvLayer{}'.format(i), ii)
				h_out = conv2d(h_out, 'ConvLayer{}'.format(i), ii)
				
				prev_channel = _conv_channel
			
			h_conv = tf.nn.relu(h_out)
#			if (is_train):
#				h_conv = tf.compat.v1.layers.dropout(h_conv, dropout_rate)
			
			#h_out_shape = np.array([h_out_shape[0] / _pool_size, h_out_shape[1] / _pool_size, _conv_channel], dtype=np.int)
			h_conv_shape = h_conv.get_shape().as_list()[1:]

			enable_bn = True
			if (enable_bn):
				bn_variables(h_conv_shape, 'ConvLayer{}'.format(i))
				h_bn = batch_norm(h_conv, 'ConvLayer{}'.format(i), is_train)
			
				h_out = max_pool(h_bn, _pool_size)
			else:
				h_out = max_pool(h_conv, _pool_size)
			h_out_shape = h_out.get_shape().as_list()[1:]
			
			prev_channel = _conv_channel
		
		# fully connected layer
		h_out = tf.reshape(h_out, [tf.shape(x)[0], -1])
		prev_channel = np.prod(h_out_shape)
		i = 0
		if (fc_channels is not None):
			for i, _fc_channel in enumerate(fc_channels):
				weight_variable([prev_channel, _fc_channel], 'FCLayer{}'.format(i))
				bias_variable([_fc_channel], 'FCLayer{}'.format(i))
				h_out = tf.nn.relu(affine(h_out, 'FCLayer{}'.format(i)))

				if (is_train):
					h_out = tf.compat.v1.layers.dropout(h_out, dropout_rate)
				prev_channel = _fc_channel
			i = i + 1
		
		weight_variable([prev_channel, output_dims[-1]], 'FCLayer{}'.format(i))
		bias_variable([output_dims[-1]], 'FCLayer{}'.format(i))
		y = affine(h_out, 'FCLayer{}'.format(i))
		
		if (is_train):
			tf.compat.v1.add_to_collection('train_input', x)
			tf.compat.v1.add_to_collection('train_output', y)
		else:
			# --- 推論側を標準名にする ---
			tf.compat.v1.add_to_collection('input', x)
			tf.compat.v1.add_to_collection('output', y)
		
		return x, y, y_
		
	def train(self, dataset,
				train_x, train_y, train_y_, 
				test_x, test_y, test_y_, 
				n_epoch=32, n_minibatch=32,
				optimizer='SGD', learning_rate=0.001,
				weight_decay=0.001,
				model_dir='model'):
		
#		weight_name = get_weight_name()
#		print(weight_name)
		
		weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_y_, logits=train_y)
		for weight in weights:
			if ('W_' in weight.name):
				print(weight.name)
				loss = loss + weight_decay * tf.nn.l2_loss(weight)
		
		if (optimizer == 'SGD'):
			train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		elif (optimizer == 'Adam'):
			train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
		else:
			print('[ERROR] unknown optimizer: {}'.format(optimizer))
			quit()
		
		correct_prediction = tf.equal(tf.argmax(train_y, 1), tf.argmax(train_y_, 1))
		train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		correct_prediction = tf.equal(tf.argmax(test_y, 1), tf.argmax(test_y_, 1))
		test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		init = tf.initialize_all_variables()
		config = tf.compat.v1.ConfigProto(
			gpu_options=tf.compat.v1.GPUOptions(
				allow_growth = True
			)
		)
		sess = tf.compat.v1.Session(config=config)
		sess.run(init)
		saver = tf.compat.v1.train.Saver()
		
		log_label = ['epoch', 'iter', 'sec per epoch', 'train_loss', 'validation_loss', 'test_loss', 'train_acc', 'validation_acc', 'test_acc', 'min_loss', 'early_stopping_counter']
		log = []
		print(log_label)
		train_data_norm = dataset.get_normalized_data('train')
		print('train data loaded')
		validation_data_norm = dataset.get_normalized_data('validation')
		print('validation data loaded')
		test_data_norm = dataset.get_normalized_data('test')
		print('test data loaded')
		iter_minibatch = len(train_data_norm) // n_minibatch
#		log_interval = iter_minibatch // 5
#		log_interval = iter_minibatch
		log_interval = 5    # [epoch]
		sec_per_epoch = []
		min_loss = 0
		epoch = 0
		early_stopping_counter = 0
		early_stopping_th = 5
#		for epoch in range(n_epoch):
		while (early_stopping_counter < early_stopping_th):
			time_epoch_start = time.time()
			for _iter in range(iter_minibatch):
				time_iter_start = time.time()
				batch_x, batch_y = dataset.next_batch(n_minibatch)
				time_nextbatch = time.time()
				sess.run(train_step, feed_dict={train_x: batch_x, train_y_: batch_y})
				time_train_step = time.time()
				
#				if ((_iter+1) % log_interval == 0):
				if ((_iter == 0) and ((epoch+1) % log_interval == 0)):
					# --- train loss/acc ---
					sep_len = len(train_data_norm) // 100
					tmp_train_loss, tmp_train_acc = [], []
					for sep in range(100):
						pos = sep * sep_len
						_loss, _acc = sess.run([loss, train_accuracy], feed_dict={train_x: train_data_norm[pos:pos+sep_len], train_y_: dataset.train_label[pos:pos+sep_len]})
						tmp_train_loss.append(np.mean(_loss))
						tmp_train_acc.append(_acc)
					time_train_loss_acc = time.time()

					if (min_loss > 0):
						if (min_loss < np.mean(tmp_train_loss)):
							early_stopping_counter += 1
						else:
							early_stopping_counter = 0
						min_loss = min(min_loss, np.mean(tmp_train_loss))
					else:
						min_loss = np.mean(tmp_train_loss)

					# --- validation loss/acc ---
					sep_len = len(validation_data_norm) // 100
					tmp_validation_loss, tmp_validation_acc = [], []
					for sep in range(100):
						pos = sep * sep_len

						_loss = sess.run(loss, feed_dict={train_x: validation_data_norm[pos:pos+sep_len], train_y_: dataset.validation_label[pos:pos+sep_len]})
						_acc = sess.run(test_accuracy, feed_dict={test_x: validation_data_norm[pos:pos+sep_len], test_y_: dataset.validation_label[pos:pos+sep_len]})

						tmp_validation_loss.append(np.mean(_loss))
						tmp_validation_acc.append(_acc)
					time_validation_loss_acc = time.time()

					# --- test loss/acc ---
					sep_len = len(test_data_norm) // 100
					tmp_test_loss, tmp_test_acc = [], []
					for sep in range(100):
						pos = sep * sep_len

						_loss = sess.run(loss, feed_dict={train_x: test_data_norm[pos:pos+sep_len], train_y_: dataset.test_label[pos:pos+sep_len]})
						_acc = sess.run(test_accuracy, feed_dict={test_x: test_data_norm[pos:pos+sep_len], test_y_: dataset.test_label[pos:pos+sep_len]})

						tmp_test_loss.append(np.mean(_loss))
						tmp_test_acc.append(_acc)
					time_test_loss_acc = time.time()
#					print(time_iter_start, time_nextbatch, time_train_step, time_train_loss_acc, time_validation_loss_acc, time_test_loss_acc)
#					quit()

#					tmp_test_loss = sess.run(loss, feed_dict={train_x: test_data_norm, train_y_: dataset.test_label})
#					tmp_test_acc = sess.run(test_accuracy, feed_dict={test_x: test_data_norm, test_y_: dataset.test_label})

					if (len(sec_per_epoch) > 0):
						log.append([epoch, _iter, np.mean(sec_per_epoch), np.mean(tmp_train_loss), np.mean(tmp_validation_loss), np.mean(tmp_test_loss), np.mean(tmp_train_acc), np.mean(tmp_validation_acc), np.mean(tmp_test_acc), min_loss, early_stopping_counter])
					else:
						log.append([epoch, _iter, 0, np.mean(tmp_train_loss), np.mean(tmp_validation_loss), np.mean(tmp_test_loss), np.mean(tmp_train_acc), np.mean(tmp_validation_acc), np.mean(tmp_test_acc), min_loss, early_stopping_counter])
					print(log[-1])
			sec_per_epoch.append(time.time() - time_epoch_start)
			epoch += 1
		
		sep_len = len(train_data_norm) // 100
		tmp_train_acc = []
		for sep in range(100):
			pos = sep * sep_len
			_acc = sess.run(train_accuracy, feed_dict={train_x: train_data_norm[pos:pos+sep_len], train_y_: dataset.train_label[pos:pos+sep_len]})
			tmp_train_acc.append(_acc)
		train_acc = np.mean(tmp_train_acc)

		sep_len = len(validation_data_norm) // 100
		tmp_test_acc = []
		for sep in range(100):
			pos = sep * sep_len
			_acc = sess.run(test_accuracy, feed_dict={test_x: validation_data_norm[pos:pos+sep_len], test_y_: dataset.validation_label[pos:pos+sep_len]})
			tmp_validation_acc.append(_acc)
		validation_acc = np.mean(tmp_validation_acc)

		sep_len = len(test_data_norm) // 100
		tmp_test_acc = []
		for sep in range(100):
			pos = sep * sep_len
			_acc = sess.run(test_accuracy, feed_dict={test_x: test_data_norm[pos:pos+sep_len], test_y_: dataset.test_label[pos:pos+sep_len]})
			tmp_test_acc.append(_acc)
		test_acc = np.mean(tmp_test_acc)
#		test_acc = sess.run(test_accuracy, feed_dict={test_x: test_data_norm, test_y_: dataset.test_label})
		
		saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
		pd.DataFrame(log).to_csv(os.path.join(model_dir, 'log.csv'), header=log_label)
		
		# --- 重みをcsvとpngで保存 ---
		for weight in weights:
			weight_dir = os.path.join(model_dir, 'weights')
			os.makedirs(weight_dir, exist_ok=True)
			
			weight_val = sess.run(weight)
			weight_name = weight.name.translate(str.maketrans({'/': '-', ':': '-'}))
			pd.DataFrame(weight_val.reshape(len(weight_val), -1)).to_csv(os.path.join(weight_dir, '{}.csv'.format(weight_name)), header=None, index=None)
			
			plt.hist(weight_val.reshape(-1), bins=32)
			plt.tight_layout()
			plt.savefig(os.path.join(weight_dir, '{}.png'.format(weight_name)))
			plt.close()
		
		sess.close()
		tf.compat.v1.reset_default_graph()
		
		return train_acc, validation_acc, test_acc
	
	def predict(self, dataset, model):
		config = tf.compat.v1.ConfigProto(
			gpu_options=tf.compat.v1.GPUOptions(
				allow_growth = True
			)
		)
		sess = tf.compat.v1.Session(config=config)
		
		saver = tf.compat.v1.train.import_meta_graph(model + '.meta', clear_devices=True)
		saver.restore(sess, model)
		
		x = tf.compat.v1.get_collection('input')[0]
		y = tf.compat.v1.get_collection('output')[0]
		prediction = sess.run(y, feed_dict={x: test_data_norm})
		
		sess.close()
		tf.reset_default_graph()
		
		return prediction
		
	def test(self, dataset, model):
		prediction = predict(dataset, model)
		
		model_dir = str(pathlib.Path(model).resolve().parent)
		compare = np.argmax(prediction, axis=1) == np.argmax(dataset.test_label, axis=1)
		accuracy = len(compare[compare==True]) / len(compare)
		result_csv = np.vstack((np.argmax(prediction, axis=1), np.argmax(dataset.test_label, axis=1))).T
		pd.DataFrame(result_csv).to_csv(os.path.join(model_dir, 'result.csv'), header=['prediction', 'labels'])
		
		return accuracy
		
	def get_ops(self, outfile):
		graph = tf.get_default_graph()
		all_ops = graph.get_operations()
		
		with open(outfile, 'w') as f:
			for _op in all_ops:
#				f.write('{}'.format(_op.op_def))
#				if ((_op.op_def.name == 'MatMul') or (_op.op_def.name == 'Add')):
#					f.write('<< {} >>\n'.format(_op.op_def.name))
#					for _input in _op.inputs:
#						f.write(' * {}\n'.format(_input))
				f.write('{}\n'.format(_op))
		
		return
	
	def get_weight_name(self):
		weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
		weight_name = []
		for weight in weights:
			print(weight.name)
			weight_name.append(weight.name)
		
		return weight_name
		
	def get_weights(self, model, outfile):
		config = tf.compat.v1.ConfigProto(
			gpu_options=tf.compat.v1.GPUOptions(
				allow_growth = True
			)
		)
		sess = tf.compat.v1.Session(config=config)
		
		saver = tf.compat.v1.train.import_meta_graph(model + '.meta', clear_devices=True)
		saver.restore(sess, model)
		
		weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
		
		np_print_th = np.get_printoptions()['threshold']
		np.set_printoptions(threshold=np.inf)
		weights_shape = []
		with open(outfile, 'w') as f:
			for _weight in tqdm.tqdm(weights):
				weight_val = sess.run(_weight)
				f.write('{}\n{}\n\n'.format(_weight, weight_val))
				if (len(weights_shape) == 0):
#					weights_shape = np.array([weight_val.shape])
					weights_shape = [weight_val.shape]
				else:
#					weights_shape = np.vstack((weights_shape, weight_val.shape))
					weights_shape.append(weight_val.shape)
		np.set_printoptions(threshold=np_print_th)
		print(weights_shape)
		
		sess.close()
		tf.reset_default_graph()
		
#		if (len(weights_shape[-1]) == 1):
#			print('output nodes = {}'.format(weights_shape[-1]))
#		else:
#			print('output nodes = {}'.format(weights_shape[-1][1]))
		
		coef = 1
		flg_detect_weight = False
		flg_detect_bias = False
		flg_no_bias = False
		stack = None
		for i, weight_shape in enumerate(weights_shape):
			if (len(weight_shape) == 1):
				coef = 2
#				print('layer{}_bias : {}'.format(i // coef, weight_shape))
				if (flg_detect_weight):
					print('layer{}_weight : {}'.format(i // coef, stack))
					print('layer{}_bias : {}'.format(i // coef, weight_shape))
					flg_detect_bias = False
					flg_detect_weight = False
				else:
					stack = weight_shape
					flg_detect_bias = True
			else:
#				print('layer{}_weight : {}'.format(i // coef, weight_shape))
				if (flg_detect_bias):
					print('layer{}_weight : {}'.format(i // coef, weight_shape))
					print('layer{}_bias : {}'.format(i // coef, stack))
					flg_detect_bias = False
					flg_detect_weight = False
				else:
					if (flg_detect_weight):
						flg_no_bias = True
						print('layer{}_weight : {}'.format((i-1) // coef, stack))
					stack = weight_shape
					flg_detect_weight = True
		
		if ((flg_no_bias) or (i == 0)):
			print('layer{}_weight : {}'.format(i // coef, stack))
		
		return


class DataLoader():
	# --- constant ---
	TYPE_CIFAR10 = 'cifar10'

	def __init__(self, dataset_type=TYPE_CIFAR10, dataset_dir=None, validation_ratio=0.1):
		'''
			dataset_type: data type('cifar10', ...(T.B.D))
			dataset_dir: dataset directory
			validation_ratio: validation data ratio against training data
		'''

		def __set_data(train_data=None, train_label=None, validation_data=None, validation_label=None, test_data=None, test_label=None):

			self.train_data = train_data
			self.train_label = train_label
			self.validation_data = validation_data
			self.validation_label = validation_label
			self.test_data = test_data
			self.test_label = test_label
			
			if (self.train_data is not None):
				self.n_train_data = len(self.train_data)
				self.idx_train_data = list(range(self.n_train_data))
			else:
				self.n_train_data = 0
				self.idx_train_data = []

			if (self.validation_data is not None):
				self.n_validation_data = len(self.validation_data)
				self.idx_validation_data = list(range(self.n_validation_data))
			else:
				self.n_train_data = 0
				self.idx_train_data = []

			if (self.test_data is not None):
				self.n_test_data = len(self.test_data)
				self.idx_test_data = list(range(self.n_test_data))
			else:
				self.n_test_data = 0
				self.idx_test_data = []
			
			return
			
		self.dataset_type = dataset_type
		if (self.dataset_type is None):
			__set_data(train_data, train_label, test_data, test_label)
			img_shape = train_data.shape[1:]
		
		elif (self.dataset_type == 'mnist'):
			print('load mnist data')
			dataset = input_data.read_data_sets(os.path.join('.', 'MNIST_data'), one_hot=True)
			__set_data(dataset.train.images, dataset.train.labels, dataset.test.images, dataset.test.labels)
			img_shape = [28, 28, 1]		# H, W, C
		elif (self.dataset_type == self.TYPE_CIFAR10):
			def unpickle(file):
				import pickle
				with open(file, 'rb') as fo:
					dict = pickle.load(fo, encoding='bytes')
				return dict
			
			identity = np.eye(10, dtype=np.int)

			# --- load train data ---
			train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
			dataset = unpickle(os.path.join(dataset_dir, train_files[0]))
			train_images_all = dataset[b'data']
			train_labels_all = [identity[i] for i in dataset[b'labels']]
			for train_file in train_files[1:]:
				dataset = unpickle(os.path.join(dataset_dir, train_file))
				train_images_all = np.vstack((train_images_all, dataset[b'data']))
				train_labels_all = np.vstack((train_labels_all, [identity[i] for i in dataset[b'labels']]))

			train_index = np.arange(len(train_images_all))
			np.random.shuffle(train_index)

			train_images = train_images_all[train_index[:int(len(train_images_all) * (1-validation_ratio))]]
			train_labels = train_labels_all[train_index[:int(len(train_labels_all) * (1-validation_ratio))]]

			validation_images = train_images_all[train_index[int(len(train_images_all) * (1-validation_ratio)):]]
			validation_labels = train_labels_all[train_index[int(len(train_labels_all) * (1-validation_ratio)):]]
			
			# --- load test data ---
			dataset = unpickle(os.path.join(dataset_dir, 'test_batch'))
			test_images = dataset[b'data']
			test_labels = np.array([identity[i] for i in dataset[b'labels']])

			# --- transpose: [N, C, H, W] -> [N, H, W, C] ---
			train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)	# N, C, H, W → N, H, W, C
			validation_images = validation_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)	# N, C, H, W → N, H, W, C
			test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)		# N, C, H, W → N, H, W, C
			
			__set_data(
				train_data=train_images, train_label=train_labels,
				validation_data = validation_images, validation_label = validation_labels, 
				test_data = test_images, test_label = test_labels)

			print('<< cifar10 data shape >>')
			print('   train data: {}'.format(train_images.shape))
			print('   validation data: {}'.format(validation_images.shape))
			print('   test data: {}'.format(test_images.shape))
			
		else:
			print('[ERROR] unknown dataset_type ... {}'.format(self.dataset_type))
			quit()
		
		return
	
	def get_normalized_data(self, data_type):
		'''
			data_type: type of data('train', 'validation', 'test')
		'''
		if (data_type == 'train'):
		    return self.train_data / 255
		elif (data_type == 'validation'):
		    return self.validation_data / 255
		else:
		    return self.test_data / 255
		
	def next_batch(self, n_minibatch):
		def random_erasing(img):
			'''
				img: image
			'''
			size = [random.randint(3, 10) for _i in range(2)]
			pos = [np.clip(random.randint(0, img.shape[i]), 0, img.shape[i]-size[i]) for i in range(2)]
			color = random.randint(0, 255)
			img_erased = img
			if (random.randint(0, 1) == 0):
				img_erased[pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1], :] = color
			
			return img_erased

		def img_scaling(img, rate):
			'''
				img: image
				rate: rate
			'''
			h, w = img.shape[:2]
			src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
			dest = src * rate
			affine = cv2.getAffineTransform(src, dest)
			affine[:2, 2] += (np.array([w, h], dtype=np.float32) * (1-rate)) / 2.0

			return cv2.warpAffine(img, affine, (w, h), cv2.INTER_LANCZOS4)

		def img_rotate(img, angle):
			'''
				img: image
				angle: angle [deg]
			'''
			h, w = img.shape[:2]
			src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
			dest = src + shifts.reshape(1, -1).astype(np.float32)
			affine = cv2.getAffineTransform(src, dest)

			return cv2.warpAffine(img, affine, (w, h))

		def img_shift(img, shifts):
			'''
				img: image
				shifts: shift [pixel]
			'''
			h, w = img.shape[:2]
			src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
			dest = src + shifts.reshape(1, -1).astype(np.float32)
			affine = cv2.getAffineTransform(src, dest)

			return cv2.warpAffine(img, affine, (w, h))


		index = random.sample(self.idx_train_data, n_minibatch)
		train_data = self.train_data[index]
		train_label = self.train_label[index]

		if (self.dataset_type == self.TYPE_CIFAR10):
		    # --- random flip ---
		    #   0: none
		    #   1: up down
		    #   2: left right
		    #   3: up down and left right
		    flip_idx = [0, 1, 2, 3]

		    # --- brightness ---
		    #   x0.8 to x1.2
		    brightness_coef = 0.2

		    # --- scaling ---
		    #   x0.8 to x1.2
		    scaling_coef = 0.4

		    # --- rotation ---
		    #   -10deg to +10deg
		    rotation_coef = 10

		    # --- shift ---
		    #   -4pix to +4pix
		    shift_coef = 4

		    for i in range(n_minibatch):
			    np.random.shuffle(flip_idx)
			    brightness = random.randint(-(brightness_coef * 255), brightness_coef * 255)
			    angle = random.randint(-rotation_coef, rotation_coef)
			    shifts = np.array([random.randint(-shift_coef, shift_coef) for i in range(2)])
			    scale_rate = ((random.random()-0.5) * scaling_coef) + 1.0

			    train_data[i] = random_erasing(train_data[i].copy())
			    train_data[i] = img_scaling(train_data[i], scale_rate)
			    #train_data[i] = img_rotate(train_data[i], angle)
			    train_data[i] = img_shift(train_data[i], shifts)

			    if (flip_idx[0] == 0):
				    train_data[i] = np.clip(train_data[i] + brightness, 0, 255)
			    elif (flip_idx[0] == 1):
				    train_data[i] = np.clip(np.flipud(train_data[i]) + brightness, 0, 255)
			    elif (flip_idx[0] == 2):
				    train_data[i] = np.clip(np.fliplr(train_data[i]) + brightness, 0, 255)
			    else:
				    train_data[i] = np.clip(np.flipud(np.fliplr(train_data[i])) + brightness, 0, 255)

		return train_data / 255, train_label

#---------------------------------
# メイン処理
#---------------------------------
def main():
	args = ArgParser()
	
	if (args.flg_train):
		try:
			with open('benchmark_tensorflow_v1.yaml') as file:
				params = yaml.safe_load(file)
		except Exception as e:
			print('[ERROR] Exception occurred: benchmark_tensorflow_v1.yaml')
			quit()
		
		train_result_name = 'train_result.csv'
		train_result_header = []
		for key in sorted(params.keys()):
			if (key != 'n_conditions'):
				train_result_header.append(key)
		train_result_header.append('train accuracy')
		train_result_header.append('validation accuracy')
		train_result_header.append('test accuracy')
		train_result_data = []
		
		print(train_result_header)
		
		for idx_param in range(params['n_conditions']):
			model_dir = 'model_{:03}'.format(idx_param)
			os.makedirs(model_dir, exist_ok=True)
			
			print('--- param #{} -------------------'.format(idx_param))
			with open(os.path.join(model_dir, 'params.yaml'), 'w') as f:
				f.write('n_conditions: 1\n')
				for key in params.keys():
					if (key != 'n_conditions'):
						if (isinstance(params[key][idx_param], str)):
							print('{}: [\'{}\']'.format(key, params[key][idx_param]))
							f.write('{}: [\'{}\']\n'.format(key, params[key][idx_param]))
						else:
							print('{}: [{}]'.format(key, params[key][idx_param]))
							f.write('{}: [{}]\n'.format(key, params[key][idx_param]))
			print('-----------------------------------')
			dataset = DataLoader(dataset_type=params['dataset'][idx_param], dataset_dir=params['dataset_dir'][idx_param])
			
#			model = conv_net
			tf_model = TF_Model()
			
			print('load model')
			input_dims = np.hstack(([None], dataset.train_data.shape[1:]))
			conv_channels = params['conv_channels'][idx_param]
			conv_kernel_size = params['conv_kernel_size'][idx_param]
			pool_size = params['pool_size'][idx_param]
			fc_channels = params['fc_channels'][idx_param]
			output_dims = [None, 10]
			dropout_rate = params['dropout_rate'][idx_param]
			train_x, train_y, train_y_ = tf_model.conv_net(
							input_dims,
							conv_channels, conv_kernel_size, pool_size, 
							fc_channels,
							output_dims,
							dropout_rate,
							True)
			test_x, test_y, test_y_ = tf_model.conv_net(
							input_dims,
							conv_channels, conv_kernel_size, pool_size, 
							fc_channels,
							output_dims,
							dropout_rate,
							False)
			print('train')
			train_acc, validation_acc, test_acc = tf_model.train(dataset,
				train_x, train_y, train_y_, 
				test_x, test_y, test_y_, 
				n_epoch=params['n_epoch'][idx_param], n_minibatch=params['n_minibatch'][idx_param],
				optimizer=params['optimizer'][idx_param], learning_rate=params['learning_rate'][idx_param],
				weight_decay=params['weight_decay'][idx_param],
				model_dir=model_dir)

			print('train_acc: {}, validation_acc: {}, test_acc: {}'.format(train_acc, validation_acc, test_acc))
			
			train_result_data_tmp = []
			for key in sorted(params.keys()):
				if (key != 'n_conditions'):
					train_result_data_tmp.append(params[key][idx_param])
			train_result_data_tmp.append(train_acc)
			train_result_data_tmp.append(validation_acc)
			train_result_data_tmp.append(test_acc)
			train_result_data.append(train_result_data_tmp)
			
		pd.DataFrame(train_result_data).to_csv(train_result_name, header=train_result_header, index=None)
			
	else:
		model_dir = str(pathlib.Path(args.model).resolve().parent)
		try:
			with open(os.path.join(model_dir, 'params.yaml')) as file:
				params = yaml.safe_load(file)
		except Exception as e:
			print('[ERROR] Exception occurred: params.yaml')
			quit()
		
		dataset = DataLoader(params['dataset'][0])
		
		accuracy = test(dataset, args.model)
		print(accuracy)
		
		get_ops(os.path.join(model_dir, 'operations.txt'))
		get_weights(args.model, os.path.join(model_dir, 'weights.txt'))
			
	return

if __name__ == '__main__':
	main()

