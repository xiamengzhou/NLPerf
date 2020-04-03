import pickle as pkl
import numpy as np
import lang2vec.lang2vec as l2v
import glob
from logging import getLogger
from task_feats import task_eval_columns
import pandas as pd
import os
import sys

logger = getLogger()

def output_l2v_features(langs, task):
    feats = ["syntax_average", "phonology_average", "inventory_average", "geo", "fam"]
    d = {}
    for langss in langs:
        for lang in langss:
            if lang not in d:
                features = l2v.get_features(lang, "+".join(feats))
                d[lang] = features
    with open(os.path.join(sys.argv[1], 'data', '{}.vec.pkl'.format(task), 'wb')) as handle:
        pkl.dump(d, handle, protocol=pkl.HIGHEST_PROTOCOL)


def convert_label(df):
    return df.values.reshape(len(df))

def load_pkl_file(file):
    return pkl.load(open(file, "rb"))


def save_pkl_file(ob, file, paras=None):
    ob["paras"] = paras
    pkl.dump(ob, open(file, "wb"))


def recover(mns, sstd, test_labels):
    if sstd:
        return test_labels * sstd + mns
    else:
        return test_labels


def uriel_distance_vec(languages):
    print('...geographic')
    geographic = l2v.geographic_distance(languages)
    print('...genetic')
    genetic = l2v.genetic_distance(languages)
    print('...inventory')
    inventory = l2v.inventory_distance(languages)
    print('...syntactic')
    syntactic = l2v.syntactic_distance(languages)
    print('...phonological')
    phonological = l2v.phonological_distance(languages)
    print('...featural')
    featural = l2v.featural_distance(languages)
    uriel_features = [genetic, syntactic, featural, phonological, inventory, geographic]
    print("-".join(languages) + " done!")
    return uriel_features


def merge_csv(prefix, output):
    extension = 'csv'
    all_filenames = [i for i in glob.glob('{}*.{}'.format(prefix, extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv = combined_csv.set_index("Unnamed: 0").sort_index()
    # combined_csv.drop(columns=["Unnamed: 0"], inplace=True)
    combined_csv.to_csv(output)
    print("Output the dataframe with size {} to {}.".format(len(combined_csv), output))


def log_args(args):
    args_dict = args.__dict__
    for key in args_dict.keys():
        value = args_dict[key]
        logger.info("{}: {}".format(key, value))

def to_csv():
    lines = open("/home/mengzhox/scripts/nlppred/model_xgboost_ud.log", "r").readlines()
    index = list(set([l.split()[0] for l in lines]))
    columns = list(set([l.split()[2][:-1] for l in lines]))
    df = pd.DataFrame(columns=[f"RMSE@{i}" for i in range(5)], index=index)
    for line in lines:
        i = line.split()[0]
        c = line.split()[-2][:-1]
        df.loc[i, c] = float(line.split()[-1])
    lines = open("/home/mengzhox/scripts/nlppred/model_xgboost_ud_mean.log", "r").readlines()
    for line in lines:
        i = line.split()[0]
        c = "RMSE@0"
        df.loc[i, c] = float(line.split()[-2])
    df.to_csv("/home/mengzhox/scripts/nlppred/model_xgboost_ud.csv")

def correlation():
    df = pd.read_csv("data/data_ud.csv")[task_eval_columns("ud")]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    print(df.corr(method ='pearson'))

