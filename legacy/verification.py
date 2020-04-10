import os
from preprocess.collect_feats import get_file_pattern, build_lc_converter
import sys
import pandas as pd


def test_vocab(vocab):
    lines = open(vocab, "r").readlines()
    for i, line in enumerate(lines):
        try:
            a, b = line.split()
            b = int(b)
        except:
            print("{}:{}".format(vocab, i))
            break


def test_files(dir, src, tgt, error_file):
    src_tgt = [tgt, src] if src > tgt else [src, tgt]
    # test tok files
    for tt in ["tok", "spm5k", "tok-vocab", "spm5k-vocab"]:
        src_file = os.path.join(dir, tt, get_file_pattern(tt).format(*src_tgt, src))
        tgt_file = os.path.join(dir, tt, get_file_pattern(tt).format(*src_tgt, tgt))
        if not os.path.exists(src_file):
            error_file.write(" ".join(src_tgt) + " src_{}_ex\n".format(tt))
        elif os.stat(src_file).st_size == 0:
            error_file.write(" ".join(src_tgt) + " src_{}_0\n".format(tt))
        if not os.path.exists(tgt_file):
            error_file.write(" ".join(src_tgt) + " tgt_{}_ex\n".format(tt))
        elif os.stat(tgt_file).st_size == 0:
            error_file.write(" ".join(src_tgt) + " tgt_{}_0\n".format(tt))


def verify():
    code_dir = sys.argv[1]
    data_dir = sys.argv[2]
    data = pd.read_csv("{}/data/data_wiki_merge.csv".format(code_dir))
    converter = build_lc_converter(code_dir)
    error_file = open("{}/error_file".format(data_dir), "w")

    lang_pairs = data.iloc[:, 1:3]
    for i, (src, tgt) in enumerate(lang_pairs.values):
        if data.iloc[i, :-6].isnull().any() and src in converter and tgt in converter:
            test_files(data_dir, converter[src], converter[tgt], error_file)


if __name__ == '__main__':
    verify()
