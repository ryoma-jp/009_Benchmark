#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import sys
import tqdm
import argparse
import pathlib
import random
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

#---------------------------------
# 関数
#---------------------------------
def fc_net(input_dims=784, hidden1_dims=300, hidden2_dims=100, output_dims=10):
	x = tf.placeholder(tf.float32, [None, input_dims])
	
	W1 = tf.Variable(tf.truncated_normal([input_dims, hidden1_dims]))
	b1 = tf.Variable(tf.zeros([hidden1_dims]))
	h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
	
	W2 = tf.Variable(tf.truncated_normal([hidden1_dims, hidden2_dims]))
	b2 = tf.Variable(tf.zeros([hidden2_dims]))
	h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)
	
	W3 = tf.Variable(tf.truncated_normal([hidden2_dims, output_dims]))
	b3 = tf.Variable(tf.zeros([output_dims]))
	y = tf.add(tf.matmul(h2, W3), b3, name='output')
	
	y_ = tf.placeholder(tf.float32, [None, output_dims])
	
	return x, y, y_


def conv_net(input_dims=[None, 28, 28, 1], conv_channels=[32, 64], conv_kernel_size=[5, 3], pool_size=[2, 2], fc_channels=None, output_dims=[None, 10]):
	"""
		input_dims: 入力次元 [N, H, W, C]
		conv_channels: 畳み込み層のChannel数 [<layer1 channel>, <layer2 channel>, ...]
		conv_kernel_size: 畳み込み層のカーネルサイズ [<layer1 kernel size>, <layer2 kernel size>, ...]
		pool_size: プーリングサイズ [<layer1 pool size>, <layer2 kernel size>, ...]
		fc_channels: 全結合層のChannel数 [<layer1 channel>, <layer2 channel>, ...]
		output_dims: 出力次元
	"""
	
	def weight_variable(shape, name):
		init = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
		with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
			var = tf.get_variable(name, shape=shape, initializer=init)
		return var
	
	def bias_variable(shape, name):
		init = tf.constant_initializer([0.1])
		with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):
			var = tf.get_variable(name, shape=shape, initializer=init)
		return var
	
	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
	def max_pool(x, size):
		return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
	
	x = tf.placeholder(tf.float32, input_dims)
	y_ = tf.placeholder(tf.float32, output_dims)
	
	# convolution layer
	h_out = x
	h_out_shape = input_dims[1:]
	prev_channel = input_dims[-1]
	for i, (_conv_channel, _conv_kernel_size, _pool_size) in enumerate(zip(conv_channels, conv_kernel_size, pool_size)):
		print(_conv_channel, _conv_kernel_size, _pool_size, prev_channel)
		W_conv = weight_variable([_conv_kernel_size, _conv_kernel_size, prev_channel, _conv_channel], 'W_conv{}'.format(i))
		prev_channel = _conv_channel
		b_conv = bias_variable([_conv_channel], 'b_conv{}'.format(i))
		h_conv = tf.nn.relu(conv2d(h_out, W_conv) + b_conv)
		h_out = max_pool(h_conv, _pool_size)
		
		h_out_shape = np.array([h_out_shape[0] / _pool_size, h_out_shape[1] / _pool_size, _conv_channel], dtype=np.int)
	
	# fully connected layer
	h_out = tf.reshape(h_out, [tf.shape(x)[0], -1])
	prev_channel = np.prod(h_out_shape)
	i = 0
	if (fc_channels is not None):
		for i, _fc_channel in enumerate(fc_channels):
			W_fc = weight_variable([prev_channel, _fc_channel], 'W_fc{}'.format(i))
			prev_channel = _fc_channel
			b_fc = bias_variable([_fc_channel], 'b_fc{}'.format(i))
			h_out = tf.nn.relu(tf.matmul(h_out, W_fc) + b_fc)
	
	config = tf.ConfigProto(
		gpu_options=tf.GPUOptions(
			allow_growth = True
		)
	)
	sess = tf.Session(config=config)
	init = tf.initialize_all_variables()
	sess.run(init)
	
	W_fc = weight_variable([prev_channel, output_dims[-1]], 'W_fc{}'.format(i))
	b_fc = bias_variable([output_dims[-1]], 'b_fc{}'.format(i))
	y = tf.matmul(h_out, W_fc) + b_fc
	
	tf.add_to_collection('input', x)
	tf.add_to_collection('output', y)
	
	return x, y, y_
	
def train(dataset, x, y, y_, 
			n_epoch=32, n_minibatch=32,
			optimizer='SGD', learning_rate=0.001,
			weight_decay=0.001,
			model_dir='model'):
	
#	weight_name = get_weight_name()
#	print(weight_name)
	
	weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
	for weight in weights:
		if ('W_' in weight.name):
			print(weight.name)
			loss = loss + weight_decay * tf.nn.l2_loss(weight)
	
	if (optimizer == 'SGD'):
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	elif (optimizer == 'Adam'):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	else:
		print('[ERROR] unknown optimizer: {}'.format(optimizer))
		quit()
	
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	init = tf.initialize_all_variables()
	config = tf.ConfigProto(
		gpu_options=tf.GPUOptions(
			allow_growth = True
		)
	)
	sess = tf.Session(config=config)
	sess.run(init)
	saver = tf.train.Saver()
	
	log_label = ['epoch', 'iter', 'train_loss', 'test_loss', 'train_acc', 'test_acc']
	log = []
	print(log_label)
	iter_minibatch = len(dataset.train_data) // n_minibatch
	log_interval = iter_minibatch // 5
	for epoch in range(n_epoch):
		for _iter in range(iter_minibatch):
			batch_x, batch_y = dataset.next_batch(n_minibatch)
			sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
			
			if ((_iter+1) % log_interval == 0):
				sep_len = len(dataset.train_data) // 5
				tmp_train_loss, tmp_train_acc = [], []
				for sep in range(5):
					pos = sep * sep_len
					_loss, _acc = sess.run([loss, accuracy], feed_dict={x: dataset.train_data[pos:pos+sep_len], y_: dataset.train_label[pos:pos+sep_len]})
					tmp_train_loss.append(np.mean(_loss))
					tmp_train_acc.append(_acc)
				tmp_test_loss, tmp_test_acc = sess.run([loss, accuracy], feed_dict={x: dataset.test_data, y_: dataset.test_label})
				log.append([epoch, _iter, np.mean(tmp_train_loss), np.mean(tmp_test_loss), np.mean(tmp_train_acc), tmp_test_acc])
				print(log[-1])
	
	sep_len = len(dataset.train_data) // 5
	tmp_train_acc = []
	for sep in range(5):
		pos = sep * sep_len
		_acc = sess.run(accuracy, feed_dict={x: dataset.train_data[pos:pos+sep_len], y_: dataset.train_label[pos:pos+sep_len]})
		tmp_train_acc.append(_acc)
	train_acc = np.mean(tmp_train_acc)
	test_acc = sess.run(accuracy, feed_dict={x: dataset.test_data, y_: dataset.test_label})
	
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
	tf.reset_default_graph()
	
	return train_acc, test_acc

def predict(dataset, model):
	config = tf.ConfigProto(
		gpu_options=tf.GPUOptions(
			allow_growth = True
		)
	)
	sess = tf.Session(config=config)
	
	saver = tf.train.import_meta_graph(model + '.meta', clear_devices=True)
	saver.restore(sess, model)
	
	x = tf.get_collection('input')[0]
	y = tf.get_collection('output')[0]
	prediction = sess.run(y, feed_dict={x: dataset.test_data})
	
	sess.close()
	tf.reset_default_graph()
	
	return prediction
	
def test(dataset, model):
	prediction = predict(dataset, model)
	
	model_dir = str(pathlib.Path(model).resolve().parent)
	compare = np.argmax(prediction, axis=1) == np.argmax(dataset.test_label, axis=1)
	accuracy = len(compare[compare==True]) / len(compare)
	result_csv = np.vstack((np.argmax(prediction, axis=1), np.argmax(dataset.test_label, axis=1))).T
	pd.DataFrame(result_csv).to_csv(os.path.join(model_dir, 'result.csv'), header=['prediction', 'labels'])
	
	return accuracy
	
def get_ops(outfile):
	graph = tf.get_default_graph()
	all_ops = graph.get_operations()
	
	with open(outfile, 'w') as f:
		for _op in all_ops:
#			f.write('{}'.format(_op.op_def))
#			if ((_op.op_def.name == 'MatMul') or (_op.op_def.name == 'Add')):
#				f.write('<< {} >>\n'.format(_op.op_def.name))
#				for _input in _op.inputs:
#					f.write(' * {}\n'.format(_input))
			f.write('{}\n'.format(_op))
	
	return

def get_weight_name():
	weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	weight_name = []
	for weight in weights:
		print(weight.name)
		weight_name.append(weight.name)
	
	return weight_name
	
def get_weights(model, outfile):
	config = tf.ConfigProto(
		gpu_options=tf.GPUOptions(
			allow_growth = True
		)
	)
	sess = tf.Session(config=config)
	
	saver = tf.train.import_meta_graph(model + '.meta', clear_devices=True)
	saver.restore(sess, model)
	
	weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	
	np_print_th = np.get_printoptions()['threshold']
	np.set_printoptions(threshold=np.inf)
	weights_shape = []
	with open(outfile, 'w') as f:
		for _weight in tqdm.tqdm(weights):
			weight_val = sess.run(_weight)
			f.write('{}\n{}\n\n'.format(_weight, weight_val))
			if (len(weights_shape) == 0):
#				weights_shape = np.array([weight_val.shape])
				weights_shape = [weight_val.shape]
			else:
#				weights_shape = np.vstack((weights_shape, weight_val.shape))
				weights_shape.append(weight_val.shape)
	np.set_printoptions(threshold=np_print_th)
	print(weights_shape)
	
	sess.close()
	tf.reset_default_graph()
	
#	if (len(weights_shape[-1]) == 1):
#		print('output nodes = {}'.format(weights_shape[-1]))
#	else:
#		print('output nodes = {}'.format(weights_shape[-1][1]))
	
	coef = 1
	flg_detect_weight = False
	flg_detect_bias = False
	flg_no_bias = False
	stack = None
	for i, weight_shape in enumerate(weights_shape):
		if (len(weight_shape) == 1):
			coef = 2
#			print('layer{}_bias : {}'.format(i // coef, weight_shape))
			if (flg_detect_weight):
				print('layer{}_weight : {}'.format(i // coef, stack))
				print('layer{}_bias : {}'.format(i // coef, weight_shape))
				flg_detect_bias = False
				flg_detect_weight = False
			else:
				stack = weight_shape
				flg_detect_bias = True
		else:
#			print('layer{}_weight : {}'.format(i // coef, weight_shape))
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

def ArgParser():
	parser = argparse.ArgumentParser(description='TensorFlow Low Level APIのサンプル', formatter_class=argparse.RawTextHelpFormatter)
	
	parser.add_argument('--train', dest='flg_train', action='store_true', help='セット時，モデルを生成')
	parser.add_argument('--model', dest='model', type=str, default=None, required=False, help='TensorFlow学習済みモデルを指定')
	
	return parser.parse_args()
	
#---------------------------------
# クラス
#---------------------------------
class Dataset():
	def __init__(self, dataset_type, train_data=None, train_label=None, test_data=None, test_label=None):
		def __set_data(train_data=None, train_label=None, test_data=None, test_label=None):
			self.train_data = train_data
			self.train_label = train_label
			self.test_data = test_data
			self.test_label = test_label
			
			self.n_train_data = len(self.train_data)
			self.n_test_data = len(self.test_data)
			
			self.idx_train_data = list(range(self.n_train_data))
			self.idx_test_data = list(range(self.n_test_data))
			
			return
			
		if (dataset_type is None):
			__set_data(train_data, train_label, test_data, test_label)
			img_shape = train_data.shape[1:]
		
		elif (dataset_type == 'mnist'):
			print('load mnist data')
			dataset = input_data.read_data_sets(os.path.join('.', 'MNIST_data'), one_hot=True)
			__set_data(dataset.train.images, dataset.train.labels, dataset.test.images, dataset.test.labels)
			img_shape = [28, 28, 1]		# H, W, C
		elif (dataset_type == 'cifar10'):
			def unpickle(file):
				import pickle
				with open(file, 'rb') as fo:
					dict = pickle.load(fo, encoding='bytes')
				return dict
			
			identity = np.eye(10, dtype=np.int)
			train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
			dataset = unpickle(os.path.join('.', 'CIFAR10', 'cifar-10-batches-py', train_files[0]))
			train_images = dataset[b'data']
			train_labels = [identity[i] for i in dataset[b'labels']]
			for train_file in train_files[1:]:
				dataset = unpickle(os.path.join('.', 'CIFAR10', 'cifar-10-batches-py', train_file))
				train_images = np.vstack((train_images, dataset[b'data']))
				train_labels = np.vstack((train_labels, [identity[i] for i in dataset[b'labels']]))
			
			dataset = unpickle(os.path.join('.', 'CIFAR10', 'cifar-10-batches-py', 'test_batch'))
			test_images = dataset[b'data']
			test_labels = np.array([identity[i] for i in dataset[b'labels']])
			
			train_images = train_images.reshape(-1, 3, 32, 32) / 255
			test_images = test_images.reshape(-1, 3, 32, 32) / 255
			train_mean = np.mean(train_images, axis=(0, 2, 3))
			train_std = np.std(train_images, axis=(0, 2, 3))
			
			test_mean = np.mean(test_images, axis=(0, 2, 3))
			test_std = np.std(test_images, axis=(0, 2, 3))
			
			for i in range(3):
				train_images[:, i, :, :] = (train_images[:, i, :, :] - train_mean[i]) / train_std[i]
				test_images[:, i, :, :] = (test_images[:, i, :, :] - train_mean[i]) / train_std[i]
			test_mean = np.mean(test_images, axis=(0, 2, 3))
			test_std = np.std(test_images, axis=(0, 2, 3))
			
			train_images = train_images.transpose(0, 2, 3, 1).reshape(-1, 32*32*3)	# N, C, H, W → N, H, W, C
			test_images = test_images.transpose(0, 2, 3, 1).reshape(-1, 32*32*3)	# N, C, H, W → N, H, W, C
			
			__set_data(train_images, train_labels, test_images, test_labels)
			img_shape = [32, 32, 3]
			
		else:
			print('[ERROR] unknown dataset_type ... {}'.format(dataset_type))
			quit()
		
		is_conv_net = True
		if (is_conv_net):
			print(self.train_data.shape)
			self.train_data = np.reshape(self.train_data, np.hstack(([-1], img_shape)))
			self.test_data = np.reshape(self.test_data, np.hstack(([-1], img_shape)))
		
		return
		
	def next_batch(self, n_minibatch):
		index = random.sample(self.idx_train_data, n_minibatch)
		return self.train_data[index], self.train_label[index]

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
			dataset = Dataset(params['dataset'][idx_param])
			
			model = conv_net
			
			print('load model')
			x, y, y_ = model(input_dims = np.hstack(([None], dataset.train_data.shape[1:])))
			print('train')
			train_acc, test_acc = train(dataset, x, y, y_, 
				n_epoch=params['n_epoch'][idx_param], n_minibatch=params['n_minibatch'][idx_param],
				optimizer=params['optimizer'][idx_param], learning_rate=params['learning_rate'][idx_param],
				weight_decay=params['weight_decay'][idx_param],
				model_dir=model_dir)
			
			train_result_data_tmp = []
			for key in sorted(params.keys()):
				if (key != 'n_conditions'):
					train_result_data_tmp.append(params[key][idx_param])
			train_result_data_tmp.append(train_acc)
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
		
		dataset = Dataset(params['dataset'][0])
		
		accuracy = test(dataset, args.model)
		print(accuracy)
		
		get_ops(os.path.join(model_dir, 'operations.txt'))
		get_weights(args.model, os.path.join(model_dir, 'weights.txt'))
			
	return

if __name__ == '__main__':
	main()

