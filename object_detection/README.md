
## For Linux

<pre>
$ source run_01.sh
$ source run_02.sh

</pre>

## For Windows

### pycocotoolsの準備

1. Visual Studio Community をインストール（ https://visualstudio.microsoft.com/ja/downloads/ ）
** パッケージはPython開発を選択
2. Anaconda をインストール（ https://www.anaconda.com/products/individual ）
** Indivisual Edition をインストール
** setuptools, cython, matplotlib をインストール
*** pip install setuptols cython matplotlib
3. setup.py を実行
** setup.pyのファイルパス区切り文字を'/'から'\'に変更
** python setup.py build_ext --inplace --compiler=msvc


