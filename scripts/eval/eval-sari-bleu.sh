#!/bin/bash

# --- PATH settings ---
DATASETS_DIR=../../datasets
RESULT_DIR=../../results
UTILS_DIR=../../utils

BPE_TOKENS=16000
BART_SCALE=base
CONSTRAINT=0
EXP_NAME=SAN_SEED11
GPU_ID=0
MODEL=checkpoint_best.pt
MODE=test
while getopts b:d:g:hlm:n:p:r:tu:v OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    d ) DATASETS_DIR=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    h ) echo "Usage: $0 [-b BPE_TOKENS] [-d DATASETS_DIR] [-g GPU_ID] [-l BART_SCALE=large] [-m MODEL] [-n EXP_NAME] [-r RESULT_DIR] [-u UTILS_DIR] [-v MODE=valid]" 1>&2
        exit 1
        ;;
    l ) BART_SCALE=large
        ;;
    m ) MODEL=$OPTARG
        ;;
    n ) EXP_NAME=$OPTARG
        ;;
    r ) RESULT_DIR=$OPTARG
        ;;      
    u ) UTILS_DIR=$OPTARG
        ;;
    v ) MODE=valid
        ;;
    esac
done


# --- Download utils ---
if type easse > /dev/null 2>&1; then
    echo "[Info] easse already exists, skipping download"
else
    echo "[Info] easse not exists, start installation to ${UTILS_DIR}/easse"
    pushd $UTILS_DIR
    git clone https://github.com/feralvam/easse.git
    cd easse
    pip install -e .
    popd
fi

# reference_dir=${DATASETS_DIR}/jumanpp
reference_dir=../../resource/SNOW/T23
if [ $MODE == "valid" ]; then
    references=${reference_dir}/${MODE}.simp
else
    references=${reference_dir}/${MODE}.simp.0,${reference_dir}/${MODE}.simp.1,${reference_dir}/${MODE}.simp.2,${reference_dir}/${MODE}.simp.3,${reference_dir}/${MODE}.simp.4,${reference_dir}/${MODE}.simp.5,${reference_dir}/${MODE}.simp.6
fi

if [ $BPE_TOKENS == "BART" ]; then
    model=${RESULT_DIR}/bpeBART-${BART_SCALE}/${EXP_NAME}/checkpoints/${MODEL}
    output_dir=${RESULT_DIR}/bpeBART-${BART_SCALE}/${EXP_NAME}/${MODE}
    input_dir=${DATASETS_DIR}/SNOW/bpeBART-${BART_SCALE}
    preprocessed_dir=${input_dir}/fairseq-preprocess
else
    model=${RESULT_DIR}/bpe${BPE_TOKENS}/${EXP_NAME}/checkpoints/${MODEL}
    output_dir=${RESULT_DIR}/bpeBART-${BART_SCALE}/${EXP_NAME}/${MODE}
    input_dir=${DATASETS_DIR}/tok/bpe${BPE_TOKENS}
    preprocessed_dir=${input_dir}/fairseq-preprocess
fi

mkdir -p $output_dir

input=${input_dir}/${MODE}.comp
result=${output_dir}/result
log=${output_dir}/eval.log

if [ $BPE_TOKENS == "BART" ]; then
    tagged_input=${input}.tagged
    sed "s/^/<s> /g" $input > $tagged_input
    input=$tagged_input
fi

CUDA_VISIBLE_DEVICES=$GPU_ID fairseq-interactive $preprocessed_dir \
    --input $input \
    --path $model \
    --batch-size 64 --buffer-size 1024 \
    --nbest 1 --beam 5 --lenpen 1.0 \
    --max-len-a 1.5 --max-len-b 0 --remove-bpe "@@ " \
    > ${result}.txt
grep ^H ${result}.txt | cut -f 3- > ${result}.sys

if [ $BPE_TOKENS == "BART" ]; then
    cp ${result}.sys ${result}.sys.tmp
    cat ${result}.sys.tmp | sed 's/<<unk>>/<unk>/g' | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^[ \t]*//g' > ${result}.sys
fi

cat ${result}.sys | sed 's/ //g' > ${result}.detok.sys

echo model name : $MODEL >> $log
easse evaluate -t custom -m 'bleu,sari' \
    --refs_sents_paths $references \
    --orig_sents_path ${reference_dir}/${MODE}.comp \
    --sys_sents_path ${result}.detok.sys \
    --tokenizer ja-mecab \
    2>&1 | tee -a $log