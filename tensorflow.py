#! -*- coding: utf-8 -*-

"""
  [tensorflow]
    python tensorflow.py --help
    
    python tensorflow.py --param_csv benchmark.csv
"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import sys
import argparse
from argparse import RawTextHelpFormatter

from common import GetParams, DataLoader

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
	type, model_dir, data_dir = GetParams(args.param_csv)
	
	for _type, _model_dir, _data_dir in zip(type, model_dir, data_dir):
		# --- DataLoader生成 ---
		data_loader = DataLoader(_data_dir)
		
		print(data_loader.GetData())
	