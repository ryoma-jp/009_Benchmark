#! -*- coding: utf-8 -*-

"""
  [tensorflow]
    python benchmark_tensorflow.py --help
    
    python benchmark_tensorflow.py --param_csv benchmark.csv
"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import sys
import time
import tqdm
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd

from common import GetParams, DataLoader

import cv2
import tensorflow as tf

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------

"""
  関数名: ArgParser
  説明：引数を解析して値を取得する
"""
def ArgParser():
	parser = argparse.ArgumentParser(description='TensorFlowによるベンチマークスコアの計測', formatter_class=RawTextHelpFormatter)
	
	# --- 引数を追加 ---
	parser.add_argument('--param_csv', dest='param_csv', type=str, required=True, help='ベンチマーク条件を記載したパラメータファイル\n'
							'[Format] type, model_dir, data_dir\n'
							'   type: classification, ...[T.B.D]\n'
							'   model_dir: 学習済みモデルが格納されたディレクトリ\n'
							'   model_name: モデルファイル群のファイル名\n'
							'   data_dir: テストデータが格納されたディレクトリを指定')
	
	args = parser.parse_args()
	
	return args

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	# --- 引数処理 ---
	args = ArgParser()
	
	# --- パラメータ取得 ---
	type, model_dir, model_name, data_dir = GetParams(args.param_csv)
	
	f_log = open('log.csv', 'w')
	f_log.write('iter,elapsed_time[sec],inference_time[sec/100iter]\n')
	for _type, _model_dir, _model_name, _data_dir in zip(type, model_dir, model_name, data_dir):
		# --- DataLoader生成 ---
		data_loader = DataLoader(_data_dir)
		
		print(data_loader.GetData())
		data = data_loader.GetData()
		
		# --- モデルロード ---
		gd = tf.compat.v1.GraphDef.FromString(open(os.path.join(_model_dir, _model_name+'_frozen.pb'), 'rb').read())
		inp, predictions = tf.compat.v1.import_graph_def(gd, return_elements = ['input:0', 'MobilenetV2/Predictions/Reshape_1:0'])
		
		# --- 推論 ---
		img = None
		x = None
		start_time = time.time()
		for cnt, (label_id, filename) in enumerate(zip(data['label_id'], data['filename'])):
			if (img is None):
				img = np.array([cv2.imread(os.path.join(_data_dir, label_id, filename))]) / 128 - 1
			else:
				try:
					img = np.vstack((img, np.array([cv2.imread(os.path.join(_data_dir, label_id, filename)) / 128 - 1])))
				except:
					print(_data_dir)
					print(label_id)
					print(filename)
					quit()
			
			if ((cnt+1) % 100 == 0):
				print(str(time.time()-start_time) + ' : ' + str(cnt+1) + ' of ' + str(len(data)))
				pre_inference = time.time()
				with tf.compat.v1.Session(graph=inp.graph):
					prediction_val = predictions.eval(feed_dict={inp: img})
					if (x is None):
						x = prediction_val
					else:
						x = np.vstack((x, prediction_val))
				img = None
				f_log.write(str(cnt+1)+','+str(time.time()-start_time)+','+str(time.time()-pre_inference)+'\n')
				
		if ((cnt+1) % 100 > 0):
			pre_inference = time.time()
			with tf.compat.v1.Session(graph=inp.graph):
				prediction_val = predictions.eval(feed_dict={inp: img})
				if (x is None):
					x = prediction_val
				else:
					x = np.vstack((x, prediction_val))
			img = None
			f_log.write(str(cnt+1)+','+str(time.time()-start_time)+','+str(time.time()-pre_inference)+'\n')
		f_log.close()

		pd.DataFrame(np.vstack((x.argmax(axis=1), data['label_id'].values, data['filename'].values)).T).to_csv('predictions.csv', header=False, index=False)
		

