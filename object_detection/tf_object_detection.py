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
import cv2
import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
from data_loader import DataLoader

#---------------------------------
# 定数定義
#---------------------------------
MSCOCO_MINIVAL_IDS = 'mscoco_minival_ids.txt'
PREDICTED_IMG_DIR = 'predicted_img'

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
	parser.add_argument('--output_dir', dest='output_dir', type=str, default=None, required=True, \
			help='推論結果出力先ディレクトリ')

	args = parser.parse_args()

	return args

def main():
	# --- 引数処理 ---
	args = ArgParser()
	print(args.dataset_type)
	print(args.dataset_dir)
	print(args.model)
	print(args.output_dir)

	# --- Create output dir ---
	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(os.path.join(args.output_dir, PREDICTED_IMG_DIR), exist_ok=True)

	# --- Load MSCOCO minival ids ---
	mscoco_minival_ids = np.loadtxt(MSCOCO_MINIVAL_IDS, delimiter="\n", dtype=int)
	print(mscoco_minival_ids)

	# --- Load COCO dataset ---
	dataset = DataLoader(dataset_type=args.dataset_type, dataset_dir=args.dataset_dir, load_ids_test=mscoco_minival_ids)

	# --- Load Model ---
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(args.model, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		sess = tf.Session(graph=detection_graph)

	# --- input and output tensors ---
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	# --- inference ---
	for cnt, img_file_name in enumerate(dataset.test_data['image_file']):
		img_file = os.path.join(args.dataset_dir, 'val2014', img_file_name)
		print('<< img_file: {} >>'.format(img_file))
		img = cv2.imread(img_file)
		print(' * img shape: {}'.format(img.shape))
		frame = np.expand_dims(img, axis=0)
		(boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: frame})

#		print('<< boxes >>\n{}\n'.format(boxes))
#		print('<< scores >>\n{}\n'.format(scores))
#		print('<< classes >>\n{}\n'.format(classes))
#		print('<< num >>\n{}\n'.format(num))

		color = (255, 0, 0)
		for i in range(int(num[0])):
			predict_point_left_top = boxes[0][i][0:2] * img.shape[0:2]	# [y1, x1]
			predict_point_right_bottom = boxes[0][i][2:4] * img.shape[0:2]	# [y2, x1]
			predict_score = scores[0][i]
			predict_class = classes[0][i]

			img = cv2.rectangle(img,
				tuple(predict_point_left_top[::-1].astype(int)),
				tuple(predict_point_right_bottom[::-1].astype(int)), color, 3)	# [x, y]の順で指定
		cv2.imwrite(os.path.join(args.output_dir, PREDICTED_IMG_DIR, 'predict_{:06d}.png'.format(cnt)), img)

	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

