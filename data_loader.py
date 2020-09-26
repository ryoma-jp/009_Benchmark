#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import skimage.io as io
import os
import random
import tqdm
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict

#---------------------------------
# クラス
#---------------------------------
class DataLoader():
	# --- dataset type ---
	DATASET_TYPE_MNIST = 'mnist'
	DATASET_TYPE_CIFAR10 = 'cifar10'
	DATASET_TYPE_COCO2014 = 'coco2014'

	# --- data type ---
	DATA_TYPE_IMAGE = 'image'
	DATA_TYPE_DICT = 'dict'
	DATA_TYPE_USROBJ = 'user_object'	# if dataset type is None

	def __init__(self, dataset_type=DATASET_TYPE_CIFAR10, dataset_dir=None, validation_ratio=0.1,
					load_ids_train=None, load_ids_test=None,
					img_resize=(300, 300)):
		'''
			dataset_type: data type('cifar10', 'coco2014', ...(T.B.D))
			dataset_dir: dataset directory
			validation_ratio: validation data ratio against training data
		'''

		def __set_data(train_data=None, train_label=None, validation_data=None, validation_label=None, test_data=None, test_label=None):

			self.train_label = train_label
			self.validation_label = validation_label
			self.test_label = test_label
			
			if (self.data_type == self.DATA_TYPE_IMAGE):
				if (train_data is not None):
					self.train_data = train_data.astype('float32')
					self.n_train_data = len(self.train_data)
					self.idx_train_data = list(range(self.n_train_data))
					self.mean_train_data = np.mean(self.train_data, axis=(0, 1, 2, 3))
					self.std_train_data = np.std(self.train_data, axis=(0, 1, 2, 3))
					print(self.mean_train_data, self.std_train_data)
					print(np.min((self.train_data - self.mean_train_data) / (self.std_train_data + 1e-7)))
					print(np.max((self.train_data - self.mean_train_data) / (self.std_train_data + 1e-7)))
#					quit()
				else:
					self.train_data = None
					self.n_train_data = 0
					self.idx_train_data = []
					self.mean_train_data = 0
					self.std_train_data = 255

				if (validation_data is not None):
					self.validation_data = validation_data.astype('float32')
					self.n_validation_data = len(self.validation_data)
					self.idx_validation_data = list(range(self.n_validation_data))
				else:
					self.validation_data = None
					self.n_validation_data = 0
					self.idx_validation_data = []

				if (test_data is not None):
					self.test_data = test_data.astype('float32')
					self.n_test_data = len(self.test_data)
					self.idx_test_data = list(range(self.n_test_data))
				else:
					self.test_data = None
					self.n_test_data = 0
					self.idx_test_data = []
			else:
				if (train_data is not None):
					self.train_data = train_data
					self.n_train_data = len(self.train_data)
					self.idx_train_data = list(range(self.n_train_data))
				else:
					self.train_data = None
					self.n_train_data = 0
					self.idx_train_data = []

				if (validation_data is not None):
					self.validation_data = validation_data
					self.n_validation_data = len(self.validation_data)
					self.idx_validation_data = list(range(self.n_validation_data))
				else:
					self.validation_data = None
					self.n_validation_data = 0
					self.idx_validation_data = []

				if (test_data is not None):
					self.test_data = test_data
					self.n_test_data = len(self.test_data)
					self.idx_test_data = list(range(self.n_test_data))
				else:
					self.test_data = None
					self.n_test_data = 0
					self.idx_test_data = []
			
			return
			
		self.dataset_type = dataset_type
		if (self.dataset_type is None):
			self.data_type = self.DATA_TYPE_USROBJ
			__set_data(train_data, train_label, test_data, test_label)
			img_shape = train_data.shape[1:]
		
		elif (self.dataset_type == self.DATASET_TYPE_MNIST):
			print('load mnist data')
			self.data_type = DATA_TYPE_IMAGE
			dataset = input_data.read_data_sets(os.path.join('.', 'MNIST_data'), one_hot=True)
			__set_data(dataset.train.images, dataset.train.labels, dataset.test.images, dataset.test.labels)
			img_shape = [28, 28, 1]		# H, W, C

		elif (self.dataset_type == self.DATASET_TYPE_CIFAR10):
			def unpickle(file):
				import pickle
				with open(file, 'rb') as fo:
					dict = pickle.load(fo, encoding='bytes')
				return dict
			
			identity = np.eye(10, dtype=np.int)

			# --- load train data ---
			self.data_type = self.DATA_TYPE_IMAGE
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
			
		elif (self.dataset_type == self.DATASET_TYPE_COCO2014):
			from pycocotools.coco import COCO

			self.data_type = self.DATA_TYPE_DICT
			annotation_dir = os.path.join(dataset_dir, 'annotations')
			val_dir = os.path.join(dataset_dir, 'val2014')
			img_file_prefix = 'COCO_val2014_'

			# --- Initialize COCO ground truth api ---
			annFile = os.path.join(annotation_dir, 'instances_val2014.json')
			self.cocoGt = COCO(annFile)

			# --- Load Categories ---
			coco_CatIds = self.cocoGt.getCatIds()
#			cats = self.cocoGt.loadCats(self.cocoGt.getCatIds())
#			print(cats)
#			imgIds = self.cocoGt.getImgIds()
#			print(imgIds)
#			imgIds = self.cocoGt.getImgIds()
#			print(imgIds)
#			nms = [cat['name'] for cat in cats[0:10]]
#			print('COCO categories: \n{}\n'.format(' '.join(nms)))

			# --- Load Annotations ---
			coco_Imgs = self.cocoGt.loadImgs(load_ids_test)
			coco_AnnIds = self.cocoGt.getAnnIds(imgIds=load_ids_test, catIds=coco_CatIds, iscrowd=None)
			coco_Anns = self.cocoGt.loadAnns(coco_AnnIds)

#			print(coco_Anns[0:10])
#			print(coco_Anns[0].keys())

			labels = OrderedDict({'image_id': [], 'bbox': [], 'category_id': []})
			cnt = 0
			for idx in load_ids_test[0:10]:
				labels['image_id'].append(coco_Anns[cnt]['image_id'])

				bbox = []
				while (idx == coco_Anns[cnt]['image_id']):
					bbox.append(coco_Anns[cnt]['bbox'])
					labels['category_id'].append(coco_Anns[cnt]['category_id'])
					cnt += 1
				labels['bbox'].append(bbox)
#			print(labels['image_id'][0:10])
#			print(labels['bbox'][0:10])
#			print(labels['category_id'][0:10])

# --- from val2014(val_dir) ---
			# --- Load Imgs ---
			if (self.data_type == self.DATA_TYPE_DICT):
				data = {'image_file': []}
				for idx in tqdm.tqdm(load_ids_test[0:10]):
					data['image_file'].append('{}{:012}.jpg'.format(img_file_prefix, idx))

				__set_data(
					test_data = data, test_label = labels)
			else:
				# [T.B.D]
				imgs = []
				debug_save_img = False	# default
#				debug_save_img = True
				for cnt, idx in enumerate(tqdm.tqdm(load_ids_test[0:10])):
					img_file = os.path.join(val_dir, '{}{:012}.jpg'.format(img_file_prefix, idx))
					img = cv2.imread(img_file)

					if (debug_save_img):
						save_dir = 'debug_save_img'
						os.makedirs(save_dir, exist_ok=True)
						save_file = os.path.join(save_dir, '{}{:012}.jpg'.format(img_file_prefix, idx))
#						print(save_file)

						# --- draw bounding box ---
						for _i in range(len(labels['bbox'][cnt])):
							# bbox: (x, y, w, h)
							point_left_top = [int(_i) for _i in labels['bbox'][cnt][_i][0:2]]
							point_right_bottom = [int(_i) for _i in labels['bbox'][cnt][_i][2:4]]
							point_right_bottom = np.add(point_left_top, point_right_bottom)
#							print(tuple(point_left_top), tuple(point_right_bottom))
							color = (255, 0, 0)
							img = cv2.rectangle(img, tuple(point_left_top), tuple(point_right_bottom), color, 3)

						cv2.imwrite(save_file, img)

					img = cv2.resize(img, img_resize)
					imgs.append(img)
				imgs = np.array(imgs)
				print(imgs.shape)

				__set_data(
					test_data = imgs, test_label = labels)

# --- from url ---
#			# --- Load Imgs ---
#			imgs = self.cocoGt.loadImgs(load_ids_test)
#
#			# --- Show Img ---
#			img = io.imread(imgs[0]['coco_url'])
#			plt.axis('off')
#			plt.imshow(img)
#			plt.savefig('img_{}.jpg'.format(load_ids_test[0]))


		else:
			print('[ERROR] unknown dataset_type ... {}'.format(self.dataset_type))
			quit()
		
		return
	
	def _normalize_data(self, data):
		return (data - self.mean_train_data) / (self.std_train_data + 1e-7)
#		return ret_data / 255
		
	def get_normalized_data(self, data_type):
		'''
			data_type: type of data('train', 'validation', 'test')
		'''
		if (data_type == 'train'):
			return self._normalize_data(self.train_data)
		elif (data_type == 'validation'):
			return self._normalize_data(self.validation_data)
		else:
			return self._normalize_data(self.test_data)
		
	def next_batch(self, n_minibatch, da_params):
		# --- da_params ---
		# * DATASET_TYPE_CIFAR10
		#     'random flip'
		#       0: none
		#       1: up down
		#       2: left right
		#       3: up down and left right
		#
		#     'brightness'
		#       brightness rate
		#
		#     'scaling'
		#       scaling rate
		#
		#     'rotation'
		#       rotation range [deg]
		#
		#     'shift'
		#       shift pixel [pix]
		#
		#     'random erasing'
		#       random erasing size (width, height) [pix]

		def random_erasing(img, size):
			'''
				img: image
			'''
#			size = [random.randint(3, 10) for _i in range(2)]
			size = [random.randint(size[0], size[1]) for _i in range(2)]
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
		train_data = self.train_data[index].copy()
		train_label = self.train_label[index]

		if (self.dataset_type == self.DATASET_TYPE_CIFAR10):
			flip_idx = da_params['random_flip']
			brightness_coef = da_params['brightness']
			scaling_coef = da_params['scaling']
			rotation_coef = da_params['rotation']
			shift_coef = da_params['shift']
			random_erasing_size = da_params['random_erasing']

			for i in range(n_minibatch):
				np.random.shuffle(flip_idx)
				brightness = random.randint(-(brightness_coef * 255), brightness_coef * 255)
				angle = random.randint(-rotation_coef, rotation_coef)
				shifts = np.array([random.randint(-shift_coef, shift_coef) for i in range(2)])
				scale_rate = (2*(random.random()-0.5) * scaling_coef) + 1.0

				train_data[i] = random_erasing(train_data[i], random_erasing_size)
				train_data[i] = img_scaling(train_data[i], scale_rate)
				train_data[i] = img_rotate(train_data[i], angle)
				train_data[i] = img_shift(train_data[i], shifts)

				if (flip_idx[0] == 0):
					train_data[i] = np.clip(train_data[i] + brightness, 0, 255)
				elif (flip_idx[0] == 1):
					train_data[i] = np.clip(np.flipud(train_data[i]) + brightness, 0, 255)
				elif (flip_idx[0] == 2):
					train_data[i] = np.clip(np.fliplr(train_data[i]) + brightness, 0, 255)
				else:
					train_data[i] = np.clip(np.flipud(np.fliplr(train_data[i])) + brightness, 0, 255)

		return self._normalize_data(train_data), train_label

