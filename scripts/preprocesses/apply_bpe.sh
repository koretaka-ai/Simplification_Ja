#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../datasets/SNOW
UTILS_DIR=../../utils

BPE_TOKENS=16000
while getopts b:d:hu: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-u UTILS_DIR]" 1>&2
        exit 0
        ;;
    u ) UTILS_DIR=$OPTARG
        ;;
    esac
done


FASTBPE=${UTILS_DIR}/fastBPE/fast

# --- Download utils ---
mkdir -p $UTILS_DIR
pushd $UTILS_DIR
if [ -d ./fastBPE ]; then
    echo "[Info] fastBPE already exists, skipping download"
else
    echo "[Info] Cloning fastBPE repository (for BPE pre-processing)..."
    git clone https://github.com/glample/fastBPE.git
fi
if [ -f ./fastBPE/fast ]; then
    echo "[Info] fastBPE already exists, skipping install"
else
    cd ./fastBPE
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
    cd ..
    if ! [[ -f ./fastBPE/fast ]]; then
        echo "[Error] fastBPE not successfully installed, abort."
        exit 1
    fi
fi
popd

tok_path=${DATASETS_DIR}/tok
bpe_path=${tok_path}/bpe${BPE_TOKENS}
mkdir -p $bpe_path

src=comp
tgt=simp

BPE_CODE=${bpe_path}/bpe${BPE_TOKENS}
BPE_VOCAB=${bpe_path}/vocab

# --- Learn BPE ---
$FASTBPE learnbpe $BPE_TOKENS ${tok_path}/train.comp ${tok_path}/train.simp > $BPE_CODE

# --- Apply codes to train ---
for l in $src $tgt; do
    $FASTBPE applybpe ${bpe_path}/train.${l} ${tok_path}/train.${l} $BPE_CODE
done

# --- Get train vocabulary ---
$FASTBPE getvocab ${bpe_path}/train.comp ${bpe_path}/train.simp > ${BPE_VOCAB}.joined_dict

# --- Apply codes to valid ---
$FASTBPE applybpe ${bpe_path}/valid.comp ${tok_path}/valid.comp $BPE_CODE ${BPE_VOCAB}.joined_dict
$FASTBPE applybpe ${bpe_path}/valid.simp ${tok_path}/valid.simp $BPE_CODE ${BPE_VOCAB}.joined_dict

# --- Apply codes to test ---
$FASTBPE applybpe ${bpe_path}/test.comp ${tok_path}/test.comp $BPE_CODE ${BPE_VOCAB}.joined_dict
$FASTBPE applybpe ${bpe_path}/test.simp ${tok_path}/test.simp.0 $BPE_CODE ${BPE_VOCAB}.joined_dict
