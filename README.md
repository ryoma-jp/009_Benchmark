# 009_Benchmark

## CIFAR-10 画像識別

[ T.B.D ]

### 参考

* データセット
** https://www.cs.toronto.edu/~kriz/cifar.html

## ひらがな識別

ndl-lab掲載のサンプルプログラムによるChainerモデルはテスト精度が**.**%．
データセット自体は学習データとテストデータで分離されておらず，
nda-labのサンプルではデータセット全体の6/7を学習データ，1/7をテストデータとして，
学習・テストを実行している．

もう少し高精度のモデルを学習してみる．

### 参考

* データセットとサンプルコード
** https://github.com/ndl-lab/hiragana_mojigazo
*** Chainerの最新版ではvolatileが非サポートとなった為，下記のように修正
<pre>
x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                                         volatile='on')
t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                                         volatile='on')

  ↓

with chainer.using_config('enable_backprop', False):
    x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
    t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))

</pre>



