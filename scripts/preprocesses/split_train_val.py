import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_name', help="Input file data")
    parser.add_argument('output_train_file_name', default="train", help="Splited train file path")
    parser.add_argument('output_valid_file_name', default="valid", help="Splited valid file path")
    parser.add_argument('-s', '--src_lang', default="comp")
    parser.add_argument('-t', '--tgt_lang', default="simp")
    parser.add_argument('-n', '--num', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    return args

def main():
    opt = parse_args()

    with open(f"{opt.input_file_name}.{opt.src_lang}") as f:
        all_src = f.readlines()
    with open(f"{opt.input_file_name}.{opt.tgt_lang}") as f:
        all_tgt = f.readlines()

    np.random.seed(opt.seed)
    all_src = np.array(all_src, dtype=object)
    all_tgt = np.array(all_tgt, dtype=object)
    shuffled_ids = np.random.permutation(len(all_src))
    train_ids = shuffled_ids[opt.num:]
    train_src = all_src[train_ids]
    train_tgt = all_tgt[train_ids]
    valid_ids = shuffled_ids[:opt.num]
    valid_src = all_src[valid_ids]
    valid_tgt = all_tgt[valid_ids]

    with open(f"{opt.output_train_file_name}.{opt.src_lang}", mode='w', encoding='utf-8') as f:
        f.writelines(list(train_src))
    with open(f"{opt.output_train_file_name}.{opt.tgt_lang}", mode='w', encoding='utf-8') as f:
        f.writelines(list(train_tgt))
    with open(f"{opt.output_valid_file_name}.{opt.src_lang}", mode='w', encoding='utf-8') as f:
        f.writelines(list(valid_src))
    with open(f"{opt.output_valid_file_name}.{opt.tgt_lang}", mode='w', encoding='utf-8') as f:
        f.writelines(list(valid_tgt))

if __name__ == '__main__':
    main()