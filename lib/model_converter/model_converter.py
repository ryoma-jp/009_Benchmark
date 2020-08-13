#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import sys
from pathlib import Path

sys.path.append(str(Path('__file__').resolve().parent.parent.parent))
print(sys.path)

import os
import numpy as np
import pandas as pd
import argparse
import yaml

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from benchmark_tensorflow_v1 import DataLoader

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='saved model(checkpoint群)をfrozen graph(protocol buffer)，tfliteへ変換する',
				formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--saved_model_dir', dest='saved_model_dir', type=str, default=None, required=True, \
			help='学習済みモデルのパス')
	parser.add_argument('--saved_model_prefix', dest='saved_model_prefix', type=str, default=None, required=True, \
			help='学習済みモデルのプレフィックス')
	parser.add_argument('--node_name_yaml', dest='node_name_yaml', type=str, default=None, required=True, \
			help='ノード名が記載されたyamlファイル')
	parser.add_argument('--output_dir', dest='output_dir', type=str, default=None, required=True, \
			help='出力ディレクトリ')
	parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default=None, required=False, \
			help='データセットが格納されているディレクトリ（tfliteの推論用）')
	parser.add_argument('--dataset_type', dest='dataset_type', type=str, default='cifar10', required=False, \
			help='データセットの種別（tfliteの推論用）')

	args = parser.parse_args()

	return args

def saved_model_to_frozen_graph(saved_model_dir, saved_model_prefix, output_node_names, output_dir):
	input_meta_graph = os.path.join(saved_model_dir, saved_model_prefix+'.meta')
	checkpoint = os.path.join(saved_model_dir, saved_model_prefix)
	output_graph_filename = os.path.join(output_dir, 'output_graph.pb')

	input_graph = ''
	input_saver_def_path = ''
	input_binary = True
	restore_op_name = ''
	filename_tensor_name = ''
	clear_devices = False

	os.makedirs(output_dir, exist_ok=True)

	'''
		check ops name
	'''
#	config = tf.compat.v1.ConfigProto(
#		gpu_options=tf.compat.v1.GPUOptions(
#			allow_growth = True
#		)
#	)
#	sess = tf.compat.v1.Session(config=config)
#	
#	saver = tf.compat.v1.train.import_meta_graph(input_meta_graph, clear_devices=True)
#	saver.restore(sess, checkpoint)
#
#	graph = tf.get_default_graph()
#	all_ops = graph.get_operations()
#	
#	outfile = 'ops.txt'
#	with open(outfile, 'w') as f:
#		for _op in all_ops:
#			f.write('{}\n'.format(_op))

	freeze_graph.freeze_graph(
		input_graph, input_saver_def_path, input_binary, checkpoint,
		output_node_names, restore_op_name, filename_tensor_name,
		output_graph_filename, clear_devices, '', '', '', input_meta_graph)

	return output_graph_filename

def frozen_graph_to_tflite(pb_file, input_node_name, output_node_name, output_dir):
	input_arrays = [input_node_name]
	output_arrays = [output_node_name]

	converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
			pb_file, input_arrays, output_arrays)
	tflite_model = converter.convert()

	output_tflite_filename = os.path.join(output_dir, 'converted_model.tflite')
	open(output_tflite_filename, 'wb').write(tflite_model)

	return output_tflite_filename

def inference(tflite_file, input_data):
	interpreter = tf.compat.v1.lite.Interpreter(model_path=tflite_file)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	input_shape = input_details[0]['shape']
	print(input_shape)
	print(input_data.shape)
	print(input_data.dtype)

	output_data = []
	for _i, _input_data in enumerate(input_data):
		if (((_i+1) % 1000) == 0):
			print('{} of {}'.format((_i+1), len(input_data)))
		interpreter.set_tensor(input_details[0]['index'], _input_data.reshape(input_shape))
		interpreter.invoke()
		output_data.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))

	return np.array(output_data)

def main():
	# --- 引数処理 ---
	args = ArgParser()

	# --- parameters ---
	saved_model_dir = args.saved_model_dir
	saved_model_prefix = args.saved_model_prefix
	node_name_yaml = os.path.join(saved_model_dir, args.node_name_yaml)
	output_dir = args.output_dir

	try:
		with open(node_name_yaml) as f:
			node_name = yaml.safe_load(f)
	except Exception as e:
		print('[ERROR] Exception occurred: {} load failed'.format(node_name_yaml))
		quit()
	input_node_names = node_name['input_node_name']
	output_node_names = node_name['output_node_name']

	pb_file = saved_model_to_frozen_graph(saved_model_dir, saved_model_prefix, output_node_names, output_dir)
	tflite_file = frozen_graph_to_tflite(pb_file, input_node_names, output_node_names, output_dir)

	if (args.dataset_dir is not None):
		dataset = DataLoader(args.dataset_type, args.dataset_dir)
		test_data = dataset.get_normalized_data('test').astype(np.float32)
		inference_label = inference(tflite_file, test_data)
		test_label = np.argmax(dataset.test_label, axis=1)
		comp_data = (inference_label==test_label) 

		print(inference_label)
		print(test_label)
		print(len(comp_data[comp_data==True]))
		print(len(test_label))
		print('accuracy: {}'.format(len(comp_data[comp_data==True]) / len(test_label)))

	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()


