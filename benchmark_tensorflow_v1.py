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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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


def conv_net(input_dims=[None, 28, 28, 1], conv_channels=[32, 64], conv_kernel_size=[5, 3], pool_size=[2, 2], fc_channels=None, output_dims=[None, 10], is_training=True):
	"""
		input_dims: 入力次元 [N, H, W, C]
		conv_channels: 畳み込み層のChannel数 [<layer1 channel>, <layer2 channel>, ...]
		conv_kernel_size: 畳み込み層のカーネルサイズ [<layer1 kernel size>, <layer2 kernel size>, ...]
		pool_size: プーリングサイズ [<layer1 pool size>, <layer2 kernel size>, ...]
		fc_channels: 全結合層のChannel数 [<layer1 channel>, <layer2 channel>, ...]
		output_dims: 出力次元
		is_training: 学習時True, 推論時False
	"""
	
	def weight_variable(shape):
		# [T.B.D] tf.get_variableで再利用できるようにする予定
		initial = tf.truncated_normal(shape, stddev=0.1)
		print('[DEBUG] {}'.format(initial))
		return tf.Variable(initial)
	
	def bias_variable(shape):
		# [T.B.D] tf.get_variableで再利用できるようにする予定
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)
	
	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
	def max_pool(x, size):
		return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
	
	if (is_training):
		reuse = False
	else:
		reuse = True
	
	with tf.variable_scope('ConvNet', reuse=reuse):
		x = tf.placeholder(tf.float32, input_dims)
		y_ = tf.placeholder(tf.float32, output_dims)
		
		# convolution layer
		h_out = x
		h_out_shape = input_dims[1:]
		prev_channel = input_dims[-1]
		for _conv_channel, _conv_kernel_size, _pool_size in zip(conv_channels, conv_kernel_size, pool_size):
			print(_conv_channel, _conv_kernel_size, _pool_size, prev_channel)
			W_conv = weight_variable([_conv_kernel_size, _conv_kernel_size, prev_channel, _conv_channel])
			prev_channel = _conv_channel
			b_conv = bias_variable([_conv_channel])
			h_conv = tf.nn.relu(conv2d(h_out, W_conv) + b_conv)
			h_out = max_pool(h_conv, _pool_size)
			
			h_out_shape = np.array([h_out_shape[0] / _pool_size, h_out_shape[1] / _pool_size, _conv_channel], dtype=np.int)
		
		# fully connected layer
		h_out = tf.reshape(h_out, [tf.shape(x)[0], -1])
		prev_channel = np.prod(h_out_shape)
		if (fc_channels is not None):
			for _fc_channel in fc_channels:
				W_fc = weight_variable([prev_channel, _fc_channel])
				prev_channel = _fc_channel
				b_fc = bias_variable([_fc_channel])
				h_out = tf.nn.relu(tf.matmul(h_out, W_fc) + b_fc)
		
		config = tf.ConfigProto(
			gpu_options=tf.GPUOptions(
				allow_growth = True
			)
		)
		sess = tf.Session(config=config)
		init = tf.initialize_all_variables()
		sess.run(init)
		
		W_fc = weight_variable([prev_channel, output_dims[-1]])
		b_fc = bias_variable([output_dims[-1]])
		y = tf.matmul(h_out, W_fc) + b_fc
	
	return x, y, y_
	
def train(dataset, x, y, y_, n_epoch=32, n_minibatch=32, model_dir='model'):
	
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
	train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
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
	
	log_label = ['epoch', 'iter', 'loss', 'train_acc']
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
				tmp_loss, tmp_acc = [], []
				for sep in range(5):
					pos = sep * sep_len
					_loss, _acc = sess.run([loss, accuracy], feed_dict={x: dataset.train_data[pos:pos+sep_len], y_: dataset.train_label[pos:pos+sep_len]})
					tmp_loss.append(np.mean(_loss))
					tmp_acc.append(_acc)
				log.append([epoch, _iter, np.mean(tmp_loss), np.mean(tmp_acc)])
				print(log[-1])
	
	print(sess.run(accuracy, feed_dict={x: dataset.test_data, y_: dataset.test_label}))
	
	os.makedirs(model_dir, exist_ok=True)
	saver.save(sess, os.path.join(model_dir, 'model.ckpt'))
	pd.DataFrame(log).to_csv(os.path.join(model_dir, 'log.csv'), header=log_label)
	
	sess.close()
	tf.reset_default_graph()
	
	return

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

def get_weights(sess, outfile):
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
	def __init__(self, train_data, train_label, test_data, test_label):
		self.train_data = train_data
		self.train_label = train_label
		self.test_data = test_data
		self.test_label = test_label
		
		self.n_train_data = len(self.train_data)
		self.n_test_data = len(self.test_data)
		
		self.idx_train_data = list(range(self.n_train_data))
		self.idx_test_data = list(range(self.n_test_data))
		
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
		dataset_type = 'mnist'
		if (dataset_type == 'mnist'):
			print('load mnist data')
			dataset = input_data.read_data_sets(os.path.join('.', 'MNIST_data'), one_hot=True)
			dataset = Dataset(
						dataset.train.images,
						dataset.train.labels,
						dataset.test.images,
						dataset.test.labels)
			img_shape = [28, 28, 1]		# H, W, C
		else:
			prnit('[ERROR] unknown dataset_type ... {}'.format(dataset_type))
			quit()
		
		is_conv_net = True
		if (is_conv_net):
			dataset.train_data = np.reshape(dataset.train_data, np.hstack(([-1], img_shape)))
			dataset.test_data = np.reshape(dataset.test_data, np.hstack(([-1], img_shape)))
		
#		models = [fc_net, conv_net]
		models = [conv_net]
		for i, model in enumerate(models):
			print('load model')
			x, y, y_ = model(input_dims = np.hstack(([None], img_shape)))
			print('train')
			train(dataset, x, y, y_, model_dir='model_{:03}'.format(i))
	else:
		config = tf.ConfigProto(
			gpu_options=tf.GPUOptions(
				allow_growth = True
			)
		)
		sess = tf.Session(config=config)
		
		saver = tf.train.import_meta_graph(args.model + '.meta', clear_devices=True)
		saver.restore(sess, args.model)
		
		model_dir = str(pathlib.Path(args.model).resolve().parent)
		
		get_ops(os.path.join(model_dir, 'operations.txt'))
		get_weights(sess, os.path.join(model_dir, 'weights.txt'))
		
	return

if __name__ == '__main__':
	main()

