#! -*- coding: utf-8 -*-

#---------------------------------
# パス設定
#---------------------------------
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'cocoapi/PythonAPI'))

#---------------------------------
# モジュールのインポート
#---------------------------------
import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
from benchmark_tensorflow_v1 import DataLoader

#---------------------------------
# 定数定義
#---------------------------------
MSCOCO_MINIVAL_IDS = "mscoco_minival_ids.txt"

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='TensorFlowの物体検出サンプル',
				formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--dataset_type', dest='dataset_type', type=str, default=None, required=True, \
			help='データセットの種類(\'coco2014\', ...)')
	parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default=None, required=True, \
			help='データセット格納先のパス')
	parser.add_argument('--model', dest='model', type=str, default=None, required=True, \
			help='推論モデル(frozon pbファイル)')

	args = parser.parse_args()

	return args

def main():
	# --- 引数処理 ---
	args = ArgParser()
	print(args.dataset_type)
	print(args.dataset_dir)
	print(args.model)

	# --- Load MSCOCO minival ids ---
	mscoco_minival_ids = np.loadtxt(MSCOCO_MINIVAL_IDS, delimiter="\n", dtype=int)
	print(mscoco_minival_ids)

	# --- Load COCO dataset ---
	dataset = DataLoader(dataset_type=args.dataset_type, dataset_dir=args.dataset_dir, load_ids_test=mscoco_minival_ids)

	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()
