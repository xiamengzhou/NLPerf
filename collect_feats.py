import pandas as pd
from iso639 import languages
import os
import numpy as np
import sys
import pickle
from utils import uriel_distance_vec


def read_file(src_file, src_bpe_file, tgt_file, tgt_bpe_file):
    src_f = open(src_file, "r")
    src_bpe_f = open(src_bpe_file, "r")
    tgt_f = open(tgt_file, "r")
    tgt_bpe_f = open(tgt_bpe_file, "r")
    src_lines = [line.split() for line in src_f.readlines()]
    src_bpe_lines = [bpe_line.split() for bpe_line in src_bpe_f.readlines()]
    tgt_lines = [line.split() for line in tgt_f.readlines()]
    tgt_bpe_lines = [bpe_line.split() for bpe_line in tgt_bpe_f.readlines()]
    src_f.close()
    src_bpe_f.close()
    tgt_f.close()
    tgt_bpe_f.close()
    return src_lines, src_bpe_lines, tgt_lines, tgt_bpe_lines


def read_vocab(vocab):
    f = open(vocab, "r")
    lines = f.readlines()
    vo = {}
    for line in lines:
        try:
            word, freq = line.split()
            vo[word] = int(freq)
        except:
            pass
    f.close()
    return vo


def read_vocabs(src_vocab, src_bpe_vocab, tgt_vocab, tgt_bpe_vocab):
    src_vocab_ = read_vocab(src_vocab)
    tgt_vocab_ = read_vocab(tgt_vocab)
    src_bpe_vocab_ = read_vocab(src_bpe_vocab)
    tgt_bpe_vocab_ = read_vocab(tgt_bpe_vocab)
    return src_vocab_, src_bpe_vocab_, tgt_vocab_, tgt_bpe_vocab_


def get_line_size(src_lines, src_bpe_lines, tgt_lines, tgt_bpe_lines):
    src_len_lines = len(src_lines)
    src_len_bpe_lines = len(src_bpe_lines)
    tgt_len_lines = len(tgt_lines)
    tgt_len_bpe_lines = len(tgt_bpe_lines)
    assert src_len_bpe_lines == src_len_lines == tgt_len_bpe_lines == tgt_len_lines
    return src_len_lines


def get_vocab_size(src_vocab, src_bpe_vocab, tgt_vocab, tgt_bpe_vocab):
    return len(src_vocab), len(src_bpe_vocab), len(tgt_vocab), len(tgt_bpe_vocab)


def get_overlap(src_vocab, tgt_vocab):
    numerator = len(set(src_vocab.keys()).intersection(tgt_vocab.keys()))
    denominator = len(src_vocab) + len(tgt_vocab)
    return numerator / denominator


def get_token_type_ratio(vocab):
    type_lens = len(vocab)
    token_lens = sum(vocab.values())
    return type_lens / token_lens


def get_file_pattern(subdir_name):
    if subdir_name == "tok":
        return "WikiMatrix.{}-{}.txt.{}.tok"
    elif subdir_name == "spm5k":
        return "WikiMatrix.{}-{}.txt.{}.tok.spm5k"
    elif subdir_name == "v2":
        return "WikiMatrix.{}-{}.txt.{}"
    elif subdir_name == "tok-vocab":
        return "WikiMatrix.{}-{}.txt.{}.tok.vocab"
    elif subdir_name == "spm5k-vocab":
        return "WikiMatrix.{}-{}.txt.{}.tok.spm5k.vocab"
    else:
        return None


def build_index(main_dir, subdir_name, output=None):
    if output is not None and os.path.exists(output):
        return pickle.load(open(output, "rb"))
    d = {}
    for dir in os.listdir(os.path.join(main_dir, subdir_name)):
        full_dir = os.path.join(main_dir, subdir_name, dir)
        if os.path.isdir(full_dir):
            filelist = os.listdir(full_dir)
            for f in filelist:
                d[f] = full_dir
        elif dir.startswith("WikiMatrix"):
            d[dir] = os.path.join(main_dir, subdir_name)
    if output is not None and not os.path.exists(output):
        pickle.dump(d, open(output, "wb"))
    return d


def get_word_count(vocab):
    return sum(vocab.values())

def build_lc_converter(code_dir):
    f = open("{}/data/files.txt".format(code_dir)).readlines()
    d = {}
    for line in f:
        part2, part3 = line.split()
        d[part3] = part2
    return d

def process_one_record():
    if src not in lc_converter or tgt not in lc_converter:
        return None
    src_part2 = lc_converter[src]
    tgt_part2 = lc_converter[tgt]

    sorted_src_tgt = [tgt_part2, src_part2] if src_part2 > tgt_part2 else [src_part2, tgt_part2]
    src_file_name = get_file_pattern("tok").format(*sorted_src_tgt, src_part2)
    tgt_file_name = get_file_pattern("tok").format(*sorted_src_tgt, tgt_part2)
    if src_file_name not in tok_index or tgt_file_name not in tok_index:
        error_file.write(get_file_pattern("v2").format(*sorted_src_tgt, src_part2) + " " + "tok\n")
        print(sorted_src_tgt)
        return None
    src_file = os.path.join(tok_index[src_file_name], src_file_name)
    tgt_file = os.path.join(tok_index[tgt_file_name], tgt_file_name)

    src_bpe_file_name = get_file_pattern("spm5k").format(*sorted_src_tgt, src_part2)
    tgt_bpe_file_name = get_file_pattern("spm5k").format(*sorted_src_tgt, tgt_part2)
    if src_bpe_file_name not in subword_index:
        error_file.write(get_file_pattern("v2").format(*sorted_src_tgt, src_part2) + " " + "spm5k\n")
        return None
    if tgt_bpe_file_name not in subword_index:
        error_file.write(get_file_pattern("v2").format(*sorted_src_tgt, tgt_part2 + " " + "spm5k\n"))
        return None
    src_bpe_file = os.path.join(subword_index[src_bpe_file_name], src_bpe_file_name)
    tgt_bpe_file = os.path.join(subword_index[tgt_bpe_file_name], tgt_bpe_file_name)

    src_lines, src_bpe_lines, tgt_lines, tgt_bpe_lines = read_file(src_file, src_bpe_file, tgt_file, tgt_bpe_file)

    src_vocab_file = os.path.join(main_path, "tok-vocab",
                                  get_file_pattern("tok-vocab").format(*sorted_src_tgt, src_part2))
    tgt_vocab_file = os.path.join(main_path, "tok-vocab",
                                  get_file_pattern("tok-vocab").format(*sorted_src_tgt, tgt_part2))
    src_bpe_vocab_file = os.path.join(main_path, "spm5k-vocab",
                                      get_file_pattern("spm5k-vocab").format(*sorted_src_tgt, src_part2))
    tgt_bpe_vocab_file = os.path.join(main_path, "spm5k-vocab",
                                      get_file_pattern("spm5k-vocab").format(*sorted_src_tgt, tgt_part2))
    if not os.path.exists(src_vocab_file):
        error_file.write(get_file_pattern("v2").format(*sorted_src_tgt, src_part2) + " " + "vocab-tok\n")
        return None
    if not os.path.exists(tgt_vocab_file):
        error_file.write(get_file_pattern("v2").format(*sorted_src_tgt, tgt_part2) + " " + "vocab-tok\n")
        return None
    if not os.path.exists(src_bpe_vocab_file):
        error_file.write(get_file_pattern("v2").format(*sorted_src_tgt, src_part2) + " " + "vocab-spm5k\n")
        return None
    if not os.path.exists(tgt_bpe_vocab_file):
        error_file.write(get_file_pattern("v2").format(*sorted_src_tgt, tgt_part2) + " " + "vocab-spm5k\n")
        return None

    src_vocab, src_bpe_vocab, tgt_vocab, tgt_bpe_vocab = read_vocabs(src_vocab_file, src_bpe_vocab_file, tgt_vocab_file,
                                                                     tgt_bpe_vocab_file)

    index_i = i + start
    try:
        line_size = get_line_size(src_lines, src_bpe_lines, tgt_lines, tgt_bpe_lines)
        data.loc[index_i, columns[0]] = line_size

        src_ttr = get_token_type_ratio(src_vocab)
        src_bpe_ttr = get_token_type_ratio(src_bpe_vocab)
        tgt_ttr = get_token_type_ratio(tgt_vocab)
        tgt_bpe_ttr = get_token_type_ratio(tgt_bpe_vocab)
        data.loc[index_i, columns[1]] = src_ttr
        data.loc[index_i, columns[2]] = src_bpe_ttr
        data.loc[index_i, columns[3]] = tgt_ttr
        data.loc[index_i, columns[4]] = tgt_bpe_ttr

        src_len_vocab, src_len_bpe_vocab, tgt_len_vocab, tgt_len_bpe_vocab = get_vocab_size(src_vocab, src_bpe_vocab,
                                                                                            tgt_vocab, tgt_bpe_vocab)
        data.loc[index_i, columns[5]] = src_len_vocab
        data.loc[index_i, columns[6]] = src_len_bpe_vocab
        data.loc[index_i, columns[7]] = tgt_len_vocab
        data.loc[index_i, columns[8]] = tgt_len_bpe_vocab

        src_word_count, src_bpe_word_count, tgt_word_count, tgt_bpe_word_count = get_word_count(src_vocab), \
                                                                                 get_word_count(src_bpe_vocab), \
                                                                                 get_word_count(tgt_vocab), \
                                                                                 get_word_count(tgt_bpe_vocab)
        src_avg_lens, tgt_avg_lens = src_word_count / line_size, tgt_word_count / line_size

        data.loc[index_i, columns[9]] = src_avg_lens
        data.loc[index_i, columns[10]] = tgt_avg_lens
        data.loc[index_i, columns[11]] = src_word_count
        data.loc[index_i, columns[12]] = src_bpe_word_count
        data.loc[index_i, columns[13]] = tgt_word_count
        data.loc[index_i, columns[14]] = tgt_bpe_word_count
    except:
        print("FAIL: {}".format(index_i))

if __name__ == '__main__':
    # code dir
    current_dir = sys.argv[1]
    # data dir
    main_path = sys.argv[2]
    # multiprocessing
    id = int(sys.argv[3])
    # number
    try:
        num = int(sys.argv[4])
    except:
        num = None
        id = 0

    data = pd.read_csv("{}/data/data_wiki_{}.csv".format(current_dir, id), index_col=0)
    columns = ["dataset size (sent)", "Source lang word TTR", "Source lang subword TTR",
               "Target lang word TTR", "Target lang subword TTR", "Source lang vocab size", "Source lang subword vocab size",
               "Target lang vocab size", "Target lang subword vocab size", "Source lang Average Sent. Length",
               "Target lang average sent. length", "Source lang word Count", "Source lang subword Count",
               "Target lang word Count", "Target lang subword Count", "geographic", "genetic", "inventory",
               "syntactic", "phonological", "featural"]

    for c in data.columns:
        if "Unnamed" in c:
            data = data.drop(columns=c)

    # for column in columns:
    #     data[column] = np.nan

    tok_index = build_index(main_dir=main_path, subdir_name="tok", output=os.path.join(main_path, "tok", "file_index.pkl"))
    subword_index = build_index(main_dir=main_path, subdir_name="spm5k", output=os.path.join(main_path, "spm5k", "file_index.pkl"))

    error_file = open(os.path.join(main_path, "error_file_{}".format(id)), "w")
    lc_converter = build_lc_converter(current_dir)

    num = len(data) if num is None else num
    start = id * num; end = (id+1) * num
    # data = data.iloc[start:end, :]
    lang_pairs = data.iloc[:, :2]

    for i, (src, tgt) in enumerate(lang_pairs.values):
        # src_lang = languages.get(part3=src)
        # tgt_lang = languages.get(part3=tgt)
        # src_part2 = src_lang.part1
        # tgt_part2 = tgt_lang.part1
        if data.iloc[i, :-6].isnull().any():
            print("Processing {}...".format(i))
            process_one_record()
        # try:
        #     feats = uriel_distance_vec([src, tgt])
        # except:
        #     feats = [np.nan] * 6
        # for i, feat in enumerate(feats):
        #     data.loc[index_i, columns[i+15]] = feats[i]
    data.to_csv("{}/data/data_wiki_{}.csv".format(current_dir, id))

