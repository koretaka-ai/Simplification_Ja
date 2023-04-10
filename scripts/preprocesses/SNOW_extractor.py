import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_path')
    parser.add_argument('output_file_name')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    return args

def main():
    opt = parse_args()

    sheet_id = 1 if opt.test else 0
    num_tgt = 7 if opt.test else 1

    df = pd.read_excel(opt.input_file_path, sheet_name=sheet_id, index_col=0)
    comp_texts = df.iloc[:,0].values.T.tolist()
    simp_texts = df.iloc[:,1:1+num_tgt].values.T.tolist()
    
    texts = [text.strip() + '\n' for text in comp_texts]
    num_text = len(texts)
    with open(f"{opt.output_file_name}.comp", 'w') as f:
        f.writelines(texts)
    print(f"[Info] Dumped train data to {opt.output_file_name}.comp")

    for i in range(num_tgt):
        file_name = f"{opt.output_file_name}.simp"
        if opt.test:
            file_name += f".{i}"
        texts = [text.strip() + '\n' for text in simp_texts[i]]
        assert len(texts) == num_text
        with open(file_name, 'w') as f:
            f.writelines(texts)
        print(f"[Info] Dumped train data to {file_name}")


if __name__ == "__main__":
    main()
