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
import json
import numpy as np
import pandas as pd
import argparse
import tqdm

import tensorflow as tf
from data_loader import DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

def get_coco_fmt_bbox(n_boxes, boxes, classes, scores, img_width, img_height, min_score_th=0.5):
	'''
	    COCOフォーマットでBoundary Box を保存する
	    Ref: https://lijiancheng0614.github.io/2017/08/22/2017_08_22_TensorFlow-Object-Detection-API/
	'''

	bboxes = []
	for i in range(n_boxes):
		if (scores[i] > min_score_th):
			y1, x1, y2, x2 = boxes[i]
			bbox = {
				'bbox': {
					'xmin': int(x1 * img_width),
					'ymin': int(y1 * img_height),
					'xmax': int(x2 * img_width),
					'ymax': int(y2 * img_height)
				},
				'category_id': int(classes[i]),
				'score': float(scores[i])
			}
			bboxes.append(bbox)

	return bboxes

def get_coco_fmt_bbox2(image_id, n_boxes, boxes, classes, scores, img_width, img_height, min_score_th=0.5):
	'''
	    cocoapi/result/instances_val2014_fakebbox100_results.json
	'''

	bboxes = []
	for i in range(n_boxes):
		if (scores[i] > min_score_th):
			y1, x1, y2, x2 = boxes[i]
			bbox = {
				'image_id': image_id,
				'bbox': [float(x1 * img_width), float(y1 * img_height), float((x2-x1) * img_width), float((y2-y1) * img_height)],
				'category_id': int(classes[i]),
				'score': float(scores[i])
			}
			bboxes.append(bbox)

	return bboxes

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
	test_annos = []
	debug_mscoco_minival_ids = mscoco_minival_ids[0:10]
	for cnt, (minival_id, img_file_name) in enumerate(tqdm.tqdm(zip(mscoco_minival_ids, dataset.test_data['image_file']))):
#	for cnt, (minival_id, img_file_name) in enumerate(zip(debug_mscoco_minival_ids, dataset.test_data['image_file'])):
		img_file = os.path.join(args.dataset_dir, 'val2014', img_file_name)
#		print('<< img_file: {} >>'.format(img_file))
		img = cv2.imread(img_file)
#		print(' * img shape: {}'.format(img.shape))
		frame = np.expand_dims(img, axis=0)
		(boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: frame})

#		print('<< boxes >>\n{}\n'.format(boxes))
#		print('<< scores >>\n{}\n'.format(scores))
#		print('<< classes >>\n{}\n'.format(classes))
#		print('<< num >>\n{}\n'.format(num))

		if (0):
			bboxes = get_coco_fmt_bbox(int(num[0]), boxes[0], classes[0], scores[0], img.shape[1], img.shape[0])
			test_annos[int(minival_id)] = {'objects': bboxes}
		else:
			bboxes = get_coco_fmt_bbox2(int(minival_id), int(num[0]), boxes[0], classes[0], scores[0], img.shape[1], img.shape[0])
			for bbox in bboxes:
				test_annos.append(bbox)
#		print(bboxes)

		color = (255, 0, 0)
		for bbox in bboxes:
#			print(bbox['category_id'], dataset.cocoGt.loadCats(bbox['category_id']))
			category = dataset.cocoGt.loadCats(bbox['category_id'])[0]
			img = cv2.rectangle(img,
				(int(bbox['bbox'][0]), int(bbox['bbox'][1])),
				(int(bbox['bbox'][0]+bbox['bbox'][2]), int(bbox['bbox'][1]+bbox['bbox'][3])), color, 3)
			img = cv2.putText(img, category['name'], 
				(int(bbox['bbox'][0]), max(int(bbox['bbox'][1]), 25)),
				cv2.FONT_HERSHEY_PLAIN, 2, color, 2, cv2.LINE_AA)
#		cv2.imwrite(os.path.join(args.output_dir, PREDICTED_IMG_DIR, 'predict_{:06d}.png'.format(cnt)), img)
		cv2.imwrite(os.path.join(args.output_dir, PREDICTED_IMG_DIR, 'predict_{:012d}.png'.format(bbox['image_id'])), img)

	with open(os.path.join(args.output_dir, 'result.json'), 'w') as fd:
		json.dump(test_annos, fd)
#	print(test_annos)

	from pycocotools.cocoeval import COCOeval
#	result_json = os.path.join('cocoapi', 'results', 'instances_val2014_fakebbox100_results.json')
	result_json = os.path.join(args.output_dir, 'result.json')
	cocoDt = dataset.cocoGt.loadRes(result_json)
	print(cocoDt)

	cocoEval = COCOeval(dataset.cocoGt, cocoDt, 'bbox')
#	cocoEval.params.imgIds = sorted(dataset.cocoGt.getImgIds())[0:100]
#	cocoEval.params.imgIds = sorted(mscoco_minival_ids[0:10])
#	cocoEval.params.imgIds = sorted(mscoco_minival_ids)
	cocoEval.params.imgIds = sorted(debug_mscoco_minival_ids)
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

