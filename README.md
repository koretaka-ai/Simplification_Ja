# 日本語のテキスト平易化
- 使用するデータセット
  - [SNOW T15:やさしい日本語コーパス](https://www.jnlp.org/GengoHouse/snow/t15)
  - [SNOW T23:やさしい日本語拡張コーパス](https://www.jnlp.org/GengoHouse/snow/t23)
- 評価指標([easse](https://github.com/feralvam/easse)を用いて評価)
  - SARI
  - BLUE
- 環境
  - python 3.8
  - cuda 11.3

# 導入
## Git clone 
~~~
git clone https://github.com/koretaka-ai/Simplification_Ja.git
~~~
## 仮想環境構築
~~~
conda create -n simplification_ja python=3.9 -y
conda activate simplification_ja
~~~
## 必要なライブラリのインストール
- fairseq (https://github.com/utanaka2000/fairseq/blob/japanese_bart_pretrained_model/JAPANESE_BART_README.md)
~~~
pushd utils
git clone -b japanese_bart_pretrained_model https://github.com/utanaka2000/fairseq.git
cd fairseq
pip install --editable ./
popd
~~~
- torch ([cudaと合うversionをinstall](https://pytorch.org/get-started/previous-versions/))
~~~
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
- cuda path setting
~~~
export CUDA_HOME=/usr/local/cuda-11.3
~~~
- apex
~~~
pushd utils
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
popd
~~~
- その他
~~~
pip install numpy==1.23.2
pip install pandas
pip install zenhan
pip install openpyxl
pip install pyknp
pip install sentencepiece
pip install sacrebleu[ja]
~~~
- [jumanpp](https://github.com/ku-nlp/jumanpp) 

# データの準備
- `resource/SNOW` のディレクトリの中にSNOW T15,T23の excel ファイルを配置してください。
~~~
pushd scripts/preprocess

# SAN, BART-base
bash prepare_corpus.sh
bash apply_bpe.sh

# BART-large
bash prepare_corpus.sh -l

popd
~~~

# モデルの訓練
~~~
pushd scripts/train

# SAN
bash fairseq_preprocess.sh
bash train-SAN.sh -n ${実験名} -g ${GPUのID} -s ${SEED値}
# ex. bash train-SAN.sh -n SAN_SEED11 -g 0 -s 11

# BART-base
bash fairseq_preprocess.sh -b BART
bash finetune-BART.sh -n ${実験名} -g ${GPUのID} -s ${SEED値}
# ex. bash finetune-BART.sh -n BART_base_SEED11 -g 0 -s 11

# BART-large
bash fairseq_preprocess.sh -b BART -l large
bash finetune-BART.sh -n ${実験名} -g ${GPUのID} -s ${SEED値} -l large
# ex. bash finetune-BART.sh -n BART_large_SEED11 -g 0 -s 11 -l large

popd
~~~
# モデルの評価（multi reference）
~~~
pushd scripts/eval

# SAN
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID}
# ex. bash eval-sari-bleu.sh -n SAN_SEED11 -g 0

# BART-base
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART
# ex. bash eval-sari-bleu.sh -n BART_base_SEED11 -g 0 -b BART

# BART-large
bash eval-sari-bleu.sh -n ${実験名} -g ${GPUのID} -b BART -l
# ex. bash eval-sari-bleu.sh -n BART_large_SEED11 -g 0 -b BART -l

popd
~~~
 
| 結果 |  BLEU  | SARI |
| ---- | ---- | ---- |
|  transformer  |  76.161  | 63.331 |
|  BART-base  |  84.165  | 64.019 |
|  BART-large  |  86.033  | 64.621 |

