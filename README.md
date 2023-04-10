# Simplification_Ja
- 日本語のテキスト平易化のまとめ
- 使用するデータセット
  - [SNOW T15:やさしい日本語コーパス](https://www.jnlp.org/GengoHouse/snow/t15)
  - [SNOW T23:やさしい日本語拡張コーパス](https://www.jnlp.org/GengoHouse/snow/t23)
- 評価指標([easse](https://github.com/feralvam/easse)を用いて評価)
  - SARI
  - BLUE
- 環境
  - python 3.9
  - cuda 11.3

# ディレクトリ構造（初期設定：SNOWのデータを以下のように配置）

# 導入
## Git clone 
~~~
git clone https://github.com/koretaka-ai/Simplification_Ja.git
~~~
## Anacondaが入っていなかったらダウンロード
- 詳しく説明しているサイト：https://hana-shin.hatenablog.com/entry/2022/02/12/203642
~~~
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
~~~
## 仮想環境構築
~~~
conda create -n simplification_ja python=3.9 -y
conda activate simplification_ja
~~~
## 必要なライブラリのインストール
- fairseq
~~~
pushd utils
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
popd
~~~
- torch ([cudaと合うversionをinstall](https://pytorch.org/get-started/previous-versions/))
~~~
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
- cuda path setting
~~~
export CUDA_HOME=/usr/local/cuda-11.3
~~~
- apex
~~~
pushd apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
popd
~~~
- その他
~~~
pip install pandas
pip install zenhan
pip install openpyxl
pip install pyknp
pip install sentencepiece
~~~
- [jumanpp](https://github.com/ku-nlp/jumanpp) 
