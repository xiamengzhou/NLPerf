import pandas as pd
from iso639 import languages
import os
import numpy as np
import sys


def read_file(src_file, src_bpe_file, tgt_file, tgt_bpe_file):
    src_lines = open(src_file, "r").readlines()
    src_bpe_lines = open(src_bpe_file, "r").readlines()
    tgt_lines = open(tgt_file, "r").readlines()
    tgt_bpe_lines = open(tgt_bpe_file, "r").readlines()
    src_lines = [line.split() for line in src_lines]
    src_bpe_lines = [bpe_line.split() for bpe_line in src_bpe_lines]
    tgt_lines = [line.split() for line in tgt_lines]
    tgt_bpe_lines = [bpe_line.split() for bpe_line in tgt_bpe_lines]
    return src_lines, src_bpe_lines, tgt_lines, tgt_bpe_lines


def read_vocab(vocab):
    lines = open(vocab, "r").readlines()
    vo = {}
    for line in lines:
        word, freq = line.split()
        vo[word] = freq
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
        return "WikiMatrix.{}-{}.txt.{}.spm5k.vocab"
    else:
        return None


def build_index(main_dir, subdir_name):
    d = {}
    for dir in os.listdir(os.path.join(main_dir, subdir_name)):
        if os.path.isdir(dir):
            full_dir = os.path.join(main_dir, subdir_name, dir)
            filelist = os.listdir(full_dir)
            for f in filelist:
                d[f] = full_dir
    return d


def get_word_count(vocab):
    return sum(vocab.values)


if __name__ == '__main__':
    data = pd.read_csv("/Users/mengzhouxia/东东/CMU/Neubig/nlppred/data/data_wiki.csv")
    columns = ["dataset size (sent)", "Source lang word TTR", "Source lang subword TTR",
               "Target lang word TTR", "Target lang subword TTR", "Source lang vocab size", "Source lang subword vocab size",
               "Target lang vocab size", "Target lang subword vocab size", "Source lang Average Sent. Length",
               "Target lang average sent. length", "Source lang word Count", "Source lang subword Count",
               "Target lang word Count", "Target lang subword Count"]

    for column in columns:
        data[column] = np.nan

    main_path = sys.argv[1]
    tok_index = build_index(main_dir=main_path, subdir_name="tok")
    subword_index = build_index(main_dir=main_path, subdir_name="spm5k")

    error_file = open(os.path.join(main_path, "error_file"), "w")

    lang_pairs = data.iloc[:, :2]
    for i, (src, tgt) in enumerate(lang_pairs.values):
        src_lang = languages.get(part3=src)
        tgt_lang = languages.get(part3=tgt)
        src_part2 = src_lang.part2
        tgt_part2 = tgt_lang.part2

        src_file = tok_index[get_file_pattern("tok").format(src_part2, tgt_part2, src_part2)]
        tgt_file = tok_index[get_file_pattern("tok").format(src_part2, tgt_part2, tgt_part2)]
        src_bpe_file = subword_index[get_file_pattern("spm5k").format(src_part2, tgt_part2, src_part2)]
        tgt_bpe_file = subword_index[get_file_pattern("spm5k").format(src_part2, tgt_part2, tgt_part2)]
        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            error_file.write(get_file_pattern("tok").format(src_part2, tgt_part2, src_part2) + " " + "tok")
            continue
        if not os.path.exists(src_bpe_file) or not os.path.exists(tgt_bpe_file):
            error_file.write(get_file_pattern("tok").format(src_part2, tgt_part2, src_part2) + " " + "spm5k")
            continue
        src_lines, src_bpe_lines, tgt_lines, tgt_bpe_lines = read_file(src_file, src_bpe_file, tgt_file, tgt_bpe_file)

        src_vocab_file = os.path.join(main_path, "tok-vocab", get_file_pattern("tok-vocab").format(src_part2, tgt_part2, src_part2))
        tgt_vocab_file = os.path.join(main_path, "tok-vocab", get_file_pattern("tok-vocab").format(src_part2, tgt_part2, tgt_part2))
        src_bpe_vocab_file = os.path.join(main_path, "spm5k-vocab", get_file_pattern("spm5k-vocab").format(src_part2, tgt_part2, src_part2))
        tgt_bpe_vocab_file = os.path.join(main_path, "spm5k-vocab", get_file_pattern("spm5k-vocab").format(src_part2, tgt_part2, tgt_part2))
        if not os.path.exists(src_vocab_file) or not os.path.exists(tgt_vocab_file):
            error_file.write(get_file_pattern("tok").format(src_part2, tgt_part2, src_part2) + " " + "vocab-tok")
            continue
        if not os.path.exists(src_bpe_vocab_file) or not os.path.exists(tgt_bpe_vocab_file):
            error_file.write(get_file_pattern("tok").format(src_part2, tgt_part2, src_part2) + " " + "vocab-spm5k")
            continue
        src_vocab, src_bpe_vocab, tgt_vocab, tgt_bpe_vocab = read_vocabs(src_vocab_file, src_bpe_vocab_file, tgt_vocab_file, tgt_bpe_vocab_file)

        index_i = data.index(i)

        line_size = get_line_size(src_lines, src_bpe_lines, tgt_lines, tgt_bpe_lines)
        data.loc[data.index(i), columns[0]] = line_size

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
    data.to_csv("/Users/mengzhouxia/东东/CMU/Neubig/nlppred/data/data_wiki2.csv")




















