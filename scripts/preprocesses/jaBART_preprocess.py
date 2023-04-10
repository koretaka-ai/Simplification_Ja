import sys
import argparse
import zenhan
from pyknp import Juman
import sentencepiece

def juman_split(line, jumanpp):
   result = jumanpp.analysis(line)
   return ' '.join([mrph.midasi for mrph in result.mrph_list()])

def bpe_encode(line, spm):
    return ' '.join(spm.EncodeAsPieces(line.strip()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('orig_file_path')
    parser.add_argument('tok_file_path')
    parser.add_argument('bpe_file_path')
    parser.add_argument('--bpe_model', required=True)
    parser.add_argument('--bpe_dict', required=True)
    args = parser.parse_args()

    jumanpp = Juman()
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.bpe_model)
    vocabs=[]
    with open(args.bpe_dict) as f:
        for line in f:
            vocabs.append(line.strip().split()[0])
    spm.set_vocabulary(vocabs)

    tok_list = []
    bpe_list = []
    with open(args.orig_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = zenhan.h2z(line)
            line = juman_split(line, jumanpp)
            tok_list.append(line + '\n')
            line = bpe_encode(line, spm)
            bpe_list.append(line + '\n')

    with open(args.tok_file_path, 'w') as f:
        f.writelines(tok_list)
    print(f"[Info] Dumped tokenized data to {args.tok_file_path}")

    with open(args.bpe_file_path, 'w') as f:
        f.writelines(bpe_list)
    print(f"[Info] Dumped bpe data to {args.bpe_file_path}")  

        
if __name__ == '__main__':
    main()
