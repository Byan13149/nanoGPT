# pretokenize_txt_to_bin.py
import os, json
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional
import sys
import zipfile

def txt_to_bin(
    txt_path: str,
    bin_path: str,
    tokenizer_name: str = "gpt2",
    insert_eos: bool = True,
    write_meta: bool = True,
    meta_path: Optional[str] = None,
    max_length: Optional[int] = 10000000,
):

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, model_max_length=max_length)
    # GPT-2 はPAD等なし。学習用に文書境界として eos を挿入するのが定番
    eos_id = tok.eos_token_id

    # 語彙サイズに応じて dtype 自動選択（GPT-2 なら uint16 でOK）
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    Path(os.path.dirname(bin_path)).mkdir(parents=True, exist_ok=True)

    # 追記モードのバイナリ書き出し
    with open(bin_path, "wb") as fbin, open(txt_path, "r", encoding="utf-8", newline="") as fin:
        for line in fin:
            line = line.rstrip("\n")
            ids = tok.encode(line, add_special_tokens=False)
            if insert_eos and eos_id is not None:
                ids.append(eos_id)
            if ids:
                np.asarray(ids, dtype=dtype).tofile(fbin)

    if write_meta:
        meta = {
            "tokenizer_name": tokenizer_name,
            "vocab_size": tok.vocab_size,
            "eos_token_id": eos_id,
            "dtype": np.dtype(dtype).name,  # "uint16" など
        }
        meta_path = meta_path or os.path.join(os.path.dirname(bin_path), "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    data_dir = "./"  

    if os.path.exists('train.txt'):
        print('Tokenized enwik8 already exists - skipping processing')
        sys.exit()

    data = zipfile.ZipFile('enwik8.zip').read('enwik8')

    print('Length of enwik8: {}'.format(len(data)))

    num_test_chars = 5000000

    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]

    for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
        print('{} will have {} bytes'.format(fn, len(part)))
        print('- Tokenizing...')
        part_str = ' '.join([str(c) if c != ord('\n') else '\n' for c in part])
        print('- Writing...')
        f = open(fn, 'w').write(part_str)
        f = open(fn + '.raw', 'wb').write(part)

    txt_to_bin(os.path.join(data_dir, "train.txt"), os.path.join(data_dir, "train.bin"))
    txt_to_bin(os.path.join(data_dir, "valid.txt"),   os.path.join(data_dir, "valid.bin"))
    txt_to_bin(os.path.join(data_dir, "test.txt"),    os.path.join(data_dir, "test.bin"))
    print("done.")
