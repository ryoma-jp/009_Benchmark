# 009_Benchmark

## CIFAR-10 画像識別

[ T.B.D ]

### 参考

* データセット  
	* https://www.cs.toronto.edu/~kriz/cifar.html  

## ひらがな識別

ndl-lab掲載のサンプルプログラムによるChainerモデルはテスト精度が98.5%程度．  
データセット自体は学習データとテストデータで分離されておらず，
nda-labのサンプルではデータセット全体の6/7を学習データ，1/7をテストデータとして，
学習・テストを実行している．  

### Raspberry Pi 3 B+で動かす

Chainerで学習したモデル（hiragana73.model）をロードして，カメラ映像を読み込んで手書き文字を識別する．

[T.B.D]

### 参考

* データセットとサンプルコード  
	* https://github.com/ndl-lab/hiragana_mojigazo  
		* Chainerの最新版ではvolatileが非サポートとなった為，下記のように修正

			    x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),  
			                                             volatile='on')
			    t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
			                                             volatile='on')
			    
			      ↓
			    
			    with chainer.using_config('enable_backprop', False):
			        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
			        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))
		* Chainerのインストール手順（参考）
			* CPUでの動作だけなら，pip installでインストールできる  
			
				    $ pip3 install chainer
			
			* GPU対応は苦労した結果，PFN提供のDocker「chainer/chainer」にcudaをインストールして対応するのか良い線だったが，解決に至らなかった  
				
				    $ nvidia-docker run -it chainer/chainer /bin/bash
				    $ apt-get update
				    $ apt-get install wget
				    $ apt-get install software-properties-common
				    $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
				    $ mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
				    $ apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
				    $ add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
				    $ apt-get update
				    $ apt-get -y install cuda
				
				下記のエラーが解決しきれずに断念
			
				    Namespace(batchsize=100, datadir='hiragana73', epoch=20, fspec='*.png', initmodel='', resume='', testratio=0.14285714285714285)
				    load NDLKANA dataset
				    n_train=68597 n_test=11403
				    n_class=73
				    Traceback (most recent call last):
				      File "example/ndlkana/train.py", line 79, in <module>
				        cuda.get_device(gpu_device).use()
				      File "cupy/cuda/device.pyx", line 150, in cupy.cuda.device.Device.use
				      File "cupy/cuda/device.pyx", line 156, in cupy.cuda.device.Device.use
				      File "cupy/cuda/runtime.pyx", line 267, in cupy.cuda.runtime.setDevice
				      File "cupy/cuda/runtime.pyx", line 201, in cupy.cuda.runtime.check_status
				    cupy.cuda.runtime.CUDARuntimeError: cudaErrorNoDevice: no CUDA-capable device is detected
				
			* ChainerのGPU対応コード参考（chainer.cudaのimportとGPUデバイスの設定）  
				https://qiita.com/ikeyasu/items/246515375b34e9fb4846
			* Chainerのインストール手順参考  
				https://docs.chainer.org/en/stable/install.html  
				https://www.sejuku.net/blog/42488
			
* Raspberry Pi 3 B+にChainerをインストールする

	    $ pip3 install chainer

	* https://qiita.com/samacoba/items/9b75faa06151235ac7ca
	* https://karaage.hatenadiary.jp/entry/2016/07/08/073000

