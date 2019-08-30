import xgboost as xgb
import numpy as np
from deprecated import deprecated
from utils import p2f
import pandas as pd
from task_feats import task_eval_metrics, task_att
import pickle as pkl
from lang2vec import lang2vec as l2v
from copy import deepcopy
from utils import standardize_feats_df

@deprecated
def read_group(file):
    f = open(file, "r").readlines()
    group_sizes = [int(l.strip()) for l in f]
    return np.array(group_sizes)

@deprecated
def read_data():
    dtrain = xgb.DMatrix('test-data/rank.train.txt')
    dtest = xgb.DMatrix('test-data/rank.test.txt')
    train_group_sizes = read_group("test-data/rank.train.qgsize.txt")
    test_group_sizes = read_group("test-data/rank.test.qgsize.txt")
    dtrain.set_group(train_group_sizes)
    dtest.set_group(test_group_sizes)
    return dtrain, dtest


def fix_sf(sf_data):
    sf_data = sf_data.dropna(axis=1, how="all")
    for c in sf_data.columns:
        if "\r" in c:
            sf_data[c.replace("\r", " ")] = sf_data[c]
            sf_data = sf_data.drop(axis=1, labels=[c])
    for c in sf_data.columns:
        if isinstance(sf_data[c][0], str) and sf_data[c][0].endswith("%"):
            sf_data[c] = sf_data[c].apply(p2f)
    sf_data["Keywords_Precision"] = sf_data["Precision"]
    sf_data["Keywords_Recall"] = sf_data["Recall"]
    sf_data["Keywords_F1"] = sf_data["F1-score"]
    sf_data["NN_Precision"] = sf_data["Precision.1"]
    sf_data["NN_Recall"] = sf_data["Recall.1"]
    sf_data["NN_F1"] = sf_data["F1-score.1"]
    sf_data = sf_data.drop(axis=1, labels=["Family", "Precision", "Recall", "F1-score", "Precision.1", "Recall.1",
                                           "F1-score.1"])
    return sf_data


def fix_bli(bli_data):
    bli_data["MUSE"] = bli_data["MUSE (Performance)"]
    bli_data["Artetxe17"] = bli_data["Artetxe17 (Performance)"]
    bli_data["Artetxe16"] = bli_data["Artetxe16 (Performance)"]
    bli_data = bli_data.drop(axis=1,
                             labels=["MUSE (Performance)", "Artetxe17 (Performance)", "Artetxe16 (Performance)"])
    return bli_data


def convert_to_one_hot(data, pref, column_index):
    vals = np.array(data.iloc[:, column_index].drop_duplicates())
    for v in vals:
        data[pref + "_" + v] = (data.iloc[:, column_index] == v).astype(int)
    return data


def output_l2v_features(langs, task):
    feats = ["syntax_average", "phonology_average", "inventory_average", "geo", "fam"]
    d = {}
    for langss in langs:
        for lang in langss:
            if lang not in d:
                features = l2v.get_features(lang, "+".join(feats))
                d[lang] = features
    with open('{}.vec.pkl'.format(task), 'wb') as handle:
        pkl.dump(d, handle, protocol=pkl.HIGHEST_PROTOCOL)

def load_l2v_features(task):
    with open('{}.vec.pkl'.format(task), "rb") as handle:
        d = pkl.load(handle)
    return d

langvec_lens = {"syntax": 103,
                "phonology": 28,
                "inventory": 158,
                "geo": 299,
                "fam": 3718}

# data file related manipulation
def get_data_langs(task, shuffle=False):
    data = pd.read_csv("data/data_{}.csv".format(task), thousands=',')
    lang_pairs = None
    langs = None
    langvec_dict = None

    if task_att(task)[2]:
        langvec_dict = load_l2v_features(task)

    # shuffle data
    if shuffle:
        # idx = np.random.permutation(data.index)
        # data = data.reindex(idx)
        # labels = labels.reindex(idx)
        data = data.sample(frac=1)

    # eliminate empty rows and columns
    data = data.dropna(axis=1, how="all")
    data = data.dropna(axis=0, how="all")

    # get the column names and drop the evaluation metrics
    headers = data.columns

    if task == "sf":
        data = convert_to_one_hot(data, "fam", 3)
        data = fix_sf(data)
        langs = data.iloc[:, 1:2]
        data = data.drop(axis=1, labels=headers[:3])
    elif task.startswith("tsf"):
        # should add target language
        data = convert_to_one_hot(data, "src", 0)
        data = convert_to_one_hot(data, "tsf", 1)
        data = data.dropna(axis=0, how="any")
        lang_pairs = data.iloc[:, 0:2]
        data = data.drop(axis=1, labels=list(headers[0:2]) + list(headers[3:5]))
    elif task == "monomt":
        langs = data.iloc[:, 1:2]
        lang_pairs = data.iloc[:, 1:3]
        data = data.drop(axis=1, labels=headers[0:3])
    elif task == "bli":
        data = convert_to_one_hot(data, "src", 1)
        data = convert_to_one_hot(data, "tsf", 2)
        data = fix_bli(data)
        lang_pairs = data.iloc[:, 1:3]
        data = data.drop(axis=1, labels=headers[0:3])
    elif task == "mi":
        data = convert_to_one_hot(data, "src", 1)
        data = convert_to_one_hot(data, "tsf", 3)
        data["Accuracy"] = data["BEST SCORE (Accuracy) from SIGMORPHON"]
        data = data.drop(axis=1, labels=["BEST SCORE (Accuracy) from SIGMORPHON"])
        lang_pairs = pd.concat([data.iloc[:, 1], data.iloc[:, 3]], axis=1)
        data = data.drop(axis=1, labels=headers[0:4])
    elif task == "wiki":
        data = convert_to_one_hot(data, "Source", 0)
        data = convert_to_one_hot(data, "Target", 1)
        lang_pairs = data.iloc[:, :2]
        data = data.drop(axis=1, labels=headers[0:2])
        data = data.dropna(axis=0, how="any")

    # extract the labels and drop the labels from data
    labels = data[task_eval_metrics(task)]
    feats = data.drop(axis=1, labels=task_eval_metrics(task))

    # add extra features
    langvecs = []
    if langvec_dict is not None:
        if lang_pairs is not None:
            for src_lang, tgt_lang in lang_pairs.values:
                langvec = langvec_dict[src_lang][src_lang] + langvec_dict[tgt_lang][tgt_lang]
                # replace missing values with -1 as an indicator (I don't think it's an appropriate way of doing so for gp)
                langvec = [-1 if isinstance(a, str) else a for a in langvec]
                langvecs.append(langvec)
        langvecs = np.stack(langvecs, axis=0)
        # for i in range(langvec.shape[0]):
            # for j in range(langvec.shape[1]):
                # if type(langvecs[i][j]) == "str":

        print(langvecs.shape)
        print(len([a + "_" + str(i) for a in langvec_lens for i in range(langvec_lens[a])]))
        langvec_df = pd.DataFrame(data=langvecs,
                                  columns=[a + "_" + str(i) for a in langvec_lens for i in range(langvec_lens[a] * 2)])
        feats = pd.concat([feats, langvec_df], axis=1)
    if not task_att(task)[3]:
        labels = labels * 100
    return feats, labels, langs, lang_pairs


def remove_na(df):
    return df.dropna(axis=0)


@deprecated
def get_transfer_data_by_group(data, lang, lang_pairs, standardize=False):
    indexes = lang_pairs[lang_pairs.iloc[:, 0] == lang].index
    start = min(indexes)
    end = max(indexes) + 1
    train_data = remove_na(pd.concat([data.iloc[0:start], data.iloc[end:]]))
    test_feats, test_labels = data.iloc[start:end, 5:], data.iloc[start:end, 2:3]
    train_feats, train_labels = train_data.iloc[:, 5:], train_data.iloc[:, 2:3]
    # train_dmatrix = xgb.DMatrix(data=train_feats, label=train_labels)
    # test_dmatrix = xgb.DMatrix(data=test_feats, label=test_labels)
    # return train_dmatrix, test_dmatrix
    mns = None; sstd = None
    if standardize:
        train_feats, test_feats = standardize_feats_df(train_feats, test_feats)
        train_labels, test_labels, mns, sstd = standardize_feats_df(train_labels, test_labels, True)
    return train_feats, train_labels, test_feats, test_labels, mns, sstd

def augment(k_fold_data, train_feats, train_labels, test_feats, test_labels, *args):
    k_fold_data["train_feats"].append(train_feats)
    k_fold_data["train_labels"].append(train_labels)
    k_fold_data["test_feats"].append(test_feats)
    k_fold_data["test_labels"].append(test_labels)
    k_fold_data["train_labels_mns"].append(args[0])
    k_fold_data["train_labels_sstd"].append(args[1])
    return k_fold_data

# get K fold data from features
def get_k_fold_data(feats, labels, lang_pairs, langs, k=10, task="tsfmt", standardize=False):
    """
        data format: {metric: {"train_feats": [], "train_labels": [], "test_feats": [], "test_labels": []}}
    """
    assert len(feats) == len(labels)
    block = {"train_feats": [], "train_labels": [], "test_feats": [], "test_labels": [],
             "test_lang_pairs": [], "test_langs": [], "train_labels_mns": [], "train_labels_sstd": []}
    k_fold_data = {}
    metrics = task_eval_metrics(task)
    for i, metric in enumerate(metrics):
        metric_labels = pd.DataFrame(labels[metric], columns=[metric])
        if lang_pairs is not None:
            feats_labels_k = remove_na(pd.concat([lang_pairs, feats, metric_labels], axis=1))
            lang_pairs_k, feats_k, labels_k = feats_labels_k.iloc[:, :2], \
                                              feats_labels_k.iloc[:, 2:-1], \
                                              feats_labels_k.iloc[:, -1:]
        else:
            assert langs is not None
            feats_labels_k = remove_na(pd.concat([langs, feats, metric_labels], axis=1))
            langs_k, feats_k, labels_k = feats_labels_k.iloc[:, :1], \
                                         feats_labels_k.iloc[:, 1:], \
                                         feats_labels_k.iloc[:, -1:]
        lens = len(feats_labels_k)
        ex_per_fold = int(np.ceil(lens / k))
        for j in range(k):
            start = ex_per_fold * j
            end = ex_per_fold * (j + 1)
            if start < lens:
                if j == 0:
                    k_fold_data[metric] = deepcopy(block)
                if lang_pairs is not None:
                    test_lang_pairs_k = lang_pairs_k.iloc[start:end, :]
                    k_fold_data[metric]["test_lang_pairs"].append(test_lang_pairs_k)
                    lang_count = len(test_lang_pairs_k)
                else:
                    test_langs_k = langs_k.iloc[start:end, :]
                    k_fold_data[metric]["test_langs"].append(test_langs_k)
                    lang_count = len(test_langs_k)
                train_feats, train_labels = pd.concat([feats_k.iloc[:start, :], feats_k.iloc[end:, :]], axis=0), \
                                            pd.concat([labels_k.iloc[:start, :], labels_k.iloc[end:, :]], axis=0)
                test_feats, test_labels = feats_k.iloc[start:end, :], labels_k.iloc[start:end, :]
                assert type(train_feats) == pd.DataFrame, type(train_labels) == pd.DataFrame
                assert type(test_feats) == pd.DataFrame, type(test_labels) == pd.DataFrame
                assert len(test_feats) == len(test_labels) == lang_count
                assert len(test_labels) > 0
                mns = None; sstd = None
                if standardize:
                    train_feats, test_feats = standardize_feats_df(train_feats, test_feats)
                    train_labels, test_labels, mns, sstd = standardize_feats_df(train_labels, test_labels, True)
                augment(k_fold_data[metric], train_feats, train_labels, test_feats, test_labels, mns, sstd)
    return k_fold_data


# random_split for data
def random_split(feats, labels, lang_pairs, langs, percentage=10, task="tsfmt", standardize=False):
    """
        data format: {metric: {"train_feats": [], "train_labels": [], "test_feats": [], "test_labels": []}}
    """
    assert len(feats) == len(labels)
    block = {"train_feats": [], "train_labels": [], "test_feats": [], "test_labels": [],
             "test_lang_pairs": [], "test_langs": [], "train_labels_mns": [], "train_labels_sstd": []}
    data = {}
    metrics = task_eval_metrics(task)
    for i, metric in enumerate(metrics):
        data[metric] = deepcopy(block)
        metric_labels = pd.DataFrame(labels[metric], columns=[metric])
        if lang_pairs is not None:
            feats_labels_k = remove_na(pd.concat([lang_pairs, feats, metric_labels], axis=1))
            lang_pairs_k, feats_k, labels_k = feats_labels_k.iloc[:, :2], \
                                              feats_labels_k.iloc[:, 2:-1], \
                                              feats_labels_k.iloc[:, -1:]
        else:
            assert langs is not None
            feats_labels_k = remove_na(pd.concat([langs, feats, metric_labels], axis=1))
            langs_k, feats_k, labels_k = feats_labels_k.iloc[:, :1], \
                                         feats_labels_k.iloc[:, 1:], \
                                         feats_labels_k.iloc[:, -1:]
        lens = len(feats_labels_k)
        test_lens = lens // percentage
        if lang_pairs is not None:
            test_lang_pairs_k = lang_pairs_k.iloc[:test_lens, :]
            data[metric]["test_lang_pairs"].append(test_lang_pairs_k)
            lang_count = len(test_lang_pairs_k)
        else:
            test_langs_k = langs_k.iloc[:test_lens, :]
            data[metric]["test_langs"].append(test_langs_k)
            lang_count = len(test_langs_k)
        train_feats, train_labels = feats_k.iloc[test_lens:, :], labels_k.iloc[test_lens:, :]
        test_feats, test_labels = feats_k.iloc[:test_lens, :], labels_k.iloc[:test_lens, :]
        assert type(train_feats) == pd.DataFrame, type(train_labels) == pd.DataFrame
        assert type(test_feats) == pd.DataFrame, type(test_labels) == pd.DataFrame
        assert len(test_feats) == len(test_labels) == lang_count
        assert len(test_labels) > 0
        mns = None; sstd = None
        if standardize:
            train_feats, test_feats = standardize_feats_df(train_feats, test_feats)
            train_labels, test_labels, mns, sstd = standardize_feats_df(train_labels, test_labels, True)
        augment(data[metric], train_feats, train_labels, test_feats, test_labels, mns, sstd)
    return data
