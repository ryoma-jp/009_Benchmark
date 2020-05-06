#! -*- coding: utf-8 -*-

"""
  [common]
    ベンチマーク共通モジュール
      python3 common.py --help
      
      python3 common.py --data_dir D:\MyProjects\Projects\PreProcess\imagenet\out_images_202004300755_threads64_resize224
"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import sys
import pandas as pd

#---------------------------------
# 定数定義
#---------------------------------
FILE_LIST_NAME = 'file_list.csv'

#---------------------------------
# 関数
#---------------------------------
"""
パラメータ取得
"""
# --- コンストラクタ ---
def GetParams(param_csv):
	params = pd.read_csv(param_csv, header=None).values
	return params[:, 0], params[:, 1], params[:, 2], params[:, 3]

#---------------------------------
# クラス
#---------------------------------
	
class DataLoader():
	"""
	データ読み込み
	"""
	# --- コンストラクタ ---
	def __init__(self, data_dir):
		self.df_file_list = pd.read_csv(os.path.join(data_dir, FILE_LIST_NAME))
		
		return
	
	# --- データ全体をDataFrameで返す ---
	def GetData(self):
		return self.df_file_list

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	# --- モジュールのインポート
	import argparse
	from argparse import RawTextHelpFormatter
	
	"""
	  関数名: ArgParser
	  説明：引数を解析して値を取得する
	"""
	def ArgParser():
		parser = argparse.ArgumentParser(description='共通モジュールのテスト', formatter_class=RawTextHelpFormatter)
		
		# --- 引数を追加 ---
		parser.add_argument('--data_dir', dest='data_dir', type=str, required=False, help='テストデータが格納されたディレクトリを指定')
		
		args = parser.parse_args()
		
		return args

	# --- 引数処理 ---
	args = ArgParser()
	
	# --- DataLoaderクラス生成 ---
	data_loader = DataLoader(args.data_dir)

