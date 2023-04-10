#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../datasets
RESOURCE_DIR=../../resource
PRE_TRAINED_DIR=../../pre-trained/

BART_SCALE=base
while getopts d:hlr: OPT
do
    case $OPT in
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-d DATASETS_DIR] [-l BART_SCALE=large] [-r RESOURCE_DIR]" 1>&2
        exit 0
        ;;
    l ) BART_SCALE=large
        ;;
    r ) RESOURCE_DIR=$OPTARG
        ;;
    esac
done

EXTRACTOR=$(dirname $0)/SNOW_extractor.py
SPLITER=$(dirname $0)/split_train_val.py
PREPROCESSER=$(dirname $0)/jaBART_preprocess.py
bpe_model=${PRE_TRAINED_DIR}/japanese_bart_${BART_SCALE}_2.0/sp.model
bpe_dict=${PRE_TRAINED_DIR}/japanese_bart_${BART_SCALE}_2.0/dict.txt

orig_path=${RESOURCE_DIR}/SNOW
tok_path=${DATASETS_DIR}/SNOW/tok
bpe_path=${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}
mkdir -p ${orig_path}/T15 ${orig_path}/T23 $tok_path $bpe_path

# Japanese BART base(large) v2.0 download
pushd ${PRE_TRAINED_DIR}
if [ -d japanese_bart_${BART_SCALE}_2.0 ]; then
    echo "[Info] Japanese_BART_${BART_SCALE} already exists, skipping download"
else
    echo "[Info] Downloading Japanese BART base(large) v2.0"
    wget "http://lotus.kuee.kyoto-u.ac.jp/nl-resource/JapaneseBARTPretrainedModel/japanese_bart_${BART_SCALE}_2.0.tar.gz"
    tar -zxvf "japanese_bart_${BART_SCALE}_2.0.tar.gz"
    rm "japanese_bart_${BART_SCALE}_2.0.tar.gz"
fi
popd

# --- Preprocess T15 ---
echo "[Info] Extructing T15 data..."
python $EXTRACTOR ${orig_path}/T15-2020.1.7.xlsx ${orig_path}/T15/train

# --- Preprocess T23 ---
echo "[Info] Extructing T23 data..."
python $EXTRACTOR ${orig_path}/T23-2020.1.7.xlsx ${orig_path}/T23/train
python $EXTRACTOR ${orig_path}/T23-2020.1.7.xlsx ${orig_path}/T23/test --test

cat ${orig_path}/T15/train.comp ${orig_path}/T23/train.comp > ${orig_path}/train.comp
cat ${orig_path}/T15/train.simp ${orig_path}/T23/train.simp > ${orig_path}/train.simp

python $SPLITER ${orig_path}/train ${orig_path}/splited-train ${orig_path}/splited-valid -s comp -t simp

# --- Tokenize T15 ---
echo "[Info] Tokenizing train data"
python $PREPROCESSER ${orig_path}/splited-train.comp ${tok_path}/train.comp ${bpe_path}/train.comp --bpe_model $bpe_model --bpe_dict $bpe_dict
python $PREPROCESSER ${orig_path}/splited-train.simp ${tok_path}/train.simp ${bpe_path}/train.simp --bpe_model $bpe_model --bpe_dict $bpe_dict

echo "[Info] Tokenizing valid data"
python $PREPROCESSER ${orig_path}/splited-valid.comp ${tok_path}/valid.comp ${bpe_path}/valid.comp --bpe_model $bpe_model --bpe_dict $bpe_dict
python $PREPROCESSER ${orig_path}/splited-valid.simp ${tok_path}/valid.simp ${bpe_path}/valid.simp --bpe_model $bpe_model --bpe_dict $bpe_dict

echo "[Info] Tokenizing test data"
python $PREPROCESSER ${orig_path}/T23/test.comp ${tok_path}/test.comp ${bpe_path}/test.comp --bpe_model $bpe_model --bpe_dict $bpe_dict
for i in $(seq 0 6); do
    python $PREPROCESSER ${orig_path}/T23/test.simp.${i} ${tok_path}/test.simp.${i} ${bpe_path}/test.simp.${i} --bpe_model $bpe_model --bpe_dict $bpe_dict
done
