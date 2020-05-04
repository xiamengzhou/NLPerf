import xgboost as xgb
import numpy as np
from collections import defaultdict
from deprecated import deprecated
import pandas as pd
from task_feats import task_eval_columns, task_att
import pickle as pkl
from copy import deepcopy
import os
from logging import getLogger
import warnings

logger = getLogger()

@deprecated
def read_group(file):
    f = open(file, "r").readlines()
    group_sizes = [int(l.strip()) for l in f]
    return np.array(group_sizes)


@deprecated
def read_data_deprecated():
    dtrain = xgb.DMatrix('test-data/rank.train.txt')
    dtest = xgb.DMatrix('test-data/rank.test.txt')
    train_group_sizes = read_group("test-data/rank.train.qgsize.txt")
    test_group_sizes = read_group("test-data/rank.test.qgsize.txt")
    dtrain.set_group(train_group_sizes)
    dtest.set_group(test_group_sizes)
    return dtrain, dtest


def p2f(x):
    return float(x.strip('%')) / 100


def fix_sf(sf_data):
    # renaming and format issue
    for c in sf_data.columns:
        if "\r" in c:
            sf_data[c.replace("\r", " ")] = sf_data[c]
            sf_data = sf_data.drop(axis=1, labels=[c])
    for c in sf_data.columns:
        if isinstance(sf_data[c][0], str) and sf_data[c][0].endswith("%"):
            sf_data[c] = sf_data[c].apply(p2f)
    sf_data = sf_data.rename(columns={"Precision": "Keywords_Precision",
                                      "Recall": "Keywords_Recall",
                                      "F1-score": "Keywords_F1",
                                      "Precision.1": "NN_Precision",
                                      "Recall.1": "NN_Recall",
                                      "F1-score.1": "NN_F1"})
    return sf_data


def fix_bli(bli_data):
    bli_data = bli_data.drop(axis=1, columns=["Optimized Sinkhorn Distance (Predictor)", "MUSE (Performance)"])
    return bli_data.rename(columns={"Cycle Sinkhorn (Performance)": "Sinkhorn",
                                    "Artetxe17 (Performance)": "Artetxe17",
                                    "Artetxe16 (Performance)": "Artetxe16"})


def fix_mi(mi_data):
    return mi_data.rename(columns={"BEST SCORE (Accuracy) from SIGMORPHON": "Accuracy"})


def convert_to_one_hot(data, pref, column_index):
    vals = np.array(data.iloc[:, column_index].drop_duplicates())
    for v in vals:
        data[pref + "_" + v] = (data.iloc[:, column_index] == v).astype(int)
    return data


def load_l2v_features(task):
    with open('{}.vec.pkl'.format(task), "rb") as handle:
        d = pkl.load(handle)
    return d


langvec_lens = {"syntax": 103,
                "phonology": 28,
                "inventory": 158,
                "geo": 299,
                "fam": 3718}


def remove_noinfo_columns(df):
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    return df.drop(cols_to_drop, axis=1)


# data file related manipulation
def read_data(task,
              shuffle=False,
              folder=None,
              selected_feats=None,
              combine_models=False):
    # shuffle: whether to shuffle the data
    # folder: base folder of the project
    # selected_feats: features to be selected
    # combine_models: whether combine data from multiple models
    #
    # return a data dictionary

    data_folder = os.path.dirname(os.path.dirname(__file__)) if folder is None else folder
    data = pd.read_csv("{}/data/data_{}.csv".format(data_folder, task), thousands=',')

    eval_columns = task_eval_columns(task)

    # hack to add features or labels for certain tasks
    if task == "ma" or task == "ud":
        data["word type ratio"] = data["word num"] / data["word type"]

    if task == "ud":
        data = data.drop(axis=1, columns=["iParse", "HUJI", "ArmParser"])
        # add dependency arcs matching WALS features
        feats_ud = pd.read_csv("{}/data/UD_BERT.csv".format(data_folder))
        cc = feats_ud.columns[3:-1]
        for index in data.index:
            lang_ud = data.loc[index, "Language"].split("_")[0]
            for column in cc:
                v = feats_ud[feats_ud["ISO 2 CODE"] == lang_ud][column].values
                if len(v) == 1:
                    v = v[0]
                else:
                    v = "--"
                data.loc[index, column] = np.nan if v == "--" else float(v)

    if task == "bli":
        # add uriel syntax features
        syntax_feats = pd.read_csv("{}/data/bli_tgt_feats.csv".format(data_folder), index_col=0)
        for index in data.index:
            for c in syntax_feats.columns:
                v = syntax_feats.loc[data.loc[index, "Target Language Code"], c]
                data.loc[index, c] = np.nan if v == "--" else float(v)
                v = syntax_feats.loc[data.loc[index, "Source Language Code"], c]
                data.loc[index, c + "_2"] = np.nan if v == "--" else float(v)

    langs = None

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
        data = fix_sf(data)
        langs = data.iloc[:, 1:2]
        # Idx,ISO-639-3,Language Name,Family
        data = data.drop(axis=1, labels=headers[:3] + ["Family"])
    elif task == "tsfmt":
        langs = data.iloc[:, 0:3]
        data = data.drop(axis=1, labels=list(headers[0:3]) + list(headers[4:6]))
    elif task.startswith("tsf"):
        langs = data.iloc[:, 0:2]
        # Source/Target language, Transfer language, Rank, Accuracy Level
        data = data.drop(axis=1, labels=list(headers[0:2]) + list(headers[3:5]))
    elif task == "monomt":
        langs = data.iloc[:, 0:2]
        data = data.drop(axis=1, labels=headers[0:2])
    elif task == "bli":
        data = fix_bli(data)
        langs = data.iloc[:, 1:3]
        data = data.drop(axis=1, labels=headers[0:3])
    elif task == "mi":
        data = fix_mi(data)
        langs = pd.concat([data.iloc[:, 1], data.iloc[:, 3]], axis=1)
        data = data.drop(axis=1, labels=headers[0:4])
    elif task == "wiki":
        # TODO: check!
        data = data.dropna(axis=0, how="any")
        langs = data.iloc[:, 0:2]
        data = data.drop(axis=1, labels=headers[0:2])
    elif task == "lemma":
        langs = data.loc[:, "language"].to_frame()
        data.drop(axis=1, labels=headers[:3], inplace=True)
    elif task == "ma":
        langs = data.loc[:, "language"].to_frame()
        data.drop(axis=1, labels=headers[:3], inplace=True)
        remove_c = [c for c in headers if "freq" in c]
        data = data.drop(axis=1, labels=remove_c)
    elif task == "ud":
        langs = data.loc[:, "Language"].to_frame()
        data.drop(axis=1, labels=headers[:2], inplace=True)
        remove_c = [c for c in headers if "freq" in c]
        data = data.drop(axis=1, labels=remove_c)
    elif task == "bli2":
        langs = data.iloc[:, 1:3]
        data = data.drop(axis=1, labels=headers[0:3])
        for index in data.index:
            for column in data.columns:
                if data.loc[index, column] == "--":
                    data.loc[index, column] = np.nan
                else:
                    data.loc[index, column] = float(data.loc[index, column])
        data = data.astype(float)

    # remove columns with the same number
    data = remove_noinfo_columns(data)

    # extract the labels and drop the labels from data
    labels = data[eval_columns]
    feats = data.drop(axis=1, labels=task_eval_columns(task))
    if selected_feats is not None:
        feats = feats[selected_feats]
    if not task_att(task)[3]:
        labels = labels * 100

    # deprecated
    # add extra features
    # langvecs = []
    # if langvec_dict is not None:
    #     if lang_pairs is not None:
    #         for src_lang, tgt_lang in lang_pairs.values:
    #             langvec = langvec_dict[src_lang][src_lang] + langvec_dict[tgt_lang][tgt_lang]
    #             # replace missing values with -1 as an indicator (I don't think it's an appropriate way of doing so for gp)
    #             langvec = [-1 if isinstance(a, str) else a for a in langvec]
    #             langvecs.append(langvec)
    #     langvecs = np.stack(langvecs, axis=0)
    #
    #     print(langvecs.shape)
    #     print(len([a + "_" + str(i) for a in langvec_lens for i in range(langvec_lens[a])]))
    #     langvec_df = pd.DataFrame(data=langvecs,
    #                               columns=[a + "_" + str(i) for a in langvec_lens for i in range(langvec_lens[a] * 2)])
    #     feats = pd.concat([feats, langvec_df], axis=1)

    # per eval model

    # for displaying purpose
    org_data = {}

    for i, model in enumerate(eval_columns):
        org_data[model] = {}

        model_labels = pd.DataFrame(labels[model], columns=[model]) # same as model_labels = labels[[models]]

        feats_labels = pd.concat([langs, feats, model_labels], axis=1)
        feats_labels.dropna(axis=0, subset=model_labels.columns, inplace=True)

        feats_labels.reset_index(drop=True, inplace=True)
        langs_, feats_, labels_ = feats_labels.loc[:, langs.columns], \
                               feats_labels.loc[:, feats.columns], \
                               feats_labels.loc[:, [model]]

        org_data[model]["feats"] = feats_
        org_data[model]["labels"] = labels_
        org_data[model]["langs"] = langs_

    if combine_models:
        org_data2 = {"all": {}}
        feats = []
        labels = []
        langs = []
        for i, model in enumerate(eval_columns):
            feats.append(org_data[model]["feats"])

            # add the model categorical feature
            feats[i]["model_" + str(model)] = 1
            for j, other_model in enumerate(eval_columns):
                if i != j:
                    feats[i]["model_" + other_model] = 0

            # add each model to labels and rename
            labels.append(org_data[model]["labels"].rename(columns={model: "all"}))

            # add each group of lang to langs
            langs.append(org_data[model]["langs"])

        # concatenate experiment records from all models
        feats = pd.concat(feats).reset_index(drop=True)
        labels = pd.concat(labels).reset_index(drop=True)
        langs = pd.concat(langs).reset_index(drop=True)

        # shuffle the data
        idx = np.random.permutation(feats.index)
        org_data2["all"]["feats"] = feats.reindex(idx)
        org_data2["all"]["labels"] = labels.reindex(idx)
        org_data2["all"]["langs"] = langs.reindex(idx)
        org_data = org_data2
        logger.info(f"Combined {', '.join(eval_columns)} into all.")

    logger.info(f"Loaded {len(org_data[list(org_data.keys())[0]]['feats'])} experiment records "
                f"for {len(org_data)} models: {', '.join(org_data.keys())}")

    for model in org_data:
        logger.info(f"Loaded {len(org_data[model]['feats'])} experiment records from {model}.")
        logger.info(f"{len(org_data[model]['feats'].columns)} feats: {', '.join(org_data[model]['feats'].columns)}")
        for lang in org_data[model]["langs"].columns:
            logger.info(f"{lang}: {len(org_data[model]['langs'][lang].unique())}")
    return org_data


class Spliter():
    def __init__(self, org_data, standardize):
        self.block = {"train_feats": [], "train_labels": [], "test_feats": [], "test_labels": [],
                      "train_langs": [], "test_langs": [], "train_labels_mns": [], "train_labels_sstd": []}
        self.org_data = org_data
        self.standardize = standardize

    def run_assertion(self, train_feats, train_labels, test_feats, test_labels, lang_count):
        assert type(train_feats) == pd.DataFrame, type(train_labels) == pd.DataFrame
        assert type(test_feats) == pd.DataFrame, type(test_labels) == pd.DataFrame
        assert len(test_feats) == len(test_labels) == lang_count
        assert len(test_labels) > 0

    def standardize_data(self, train_feats, train_labels, test_feats, test_labels):
        # standardization for models like gaussian process
        mns = None
        sstd = None
        if self.standardize:
            train_feats, test_feats = self.standardize_feats_df(train_feats, test_feats)
            train_labels, test_labels, mns, sstd = self.standardize_feats_df(train_labels, test_labels, True)
        return train_feats, train_labels, test_feats, test_labels, mns, sstd

    def standardize_feats_df(self, train_feats, test_feats, return_mean_var=False):
        columns = train_feats.columns
        train_feats = deepcopy(train_feats)
        test_feats = deepcopy(test_feats)
        mns = None
        sstd = None
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            c = None
            try:
                for c in columns:
                    if list(set(train_feats[c].values)) == [0, 1]:  # a stupid way to identify dummy variables
                        pass
                    else:
                        values, mns, sstd = self.zscore(train_feats[c].values)
                        train_feats[c] = values
                        test_feats[c] = self.zscore(test_feats[c].values, mns=mns, sstd=sstd)[0]
            except Warning:
                logger.warning("Standardization goes wrong for column {} ... with value set {}".format(c, set(
                    train_feats[c].values)))
        if return_mean_var:
            assert len(columns) == 1
            return train_feats, test_feats, mns, sstd
        else:
            return train_feats, test_feats

    def zscore(self, a, axis=0, ddof=0, mns=None, sstd=None):
        a = np.asanyarray(a)
        if mns is None:
            mns = a.mean(axis=axis)
        if sstd is None:
            sstd = a.std(axis=axis, ddof=ddof)
        if axis and mns.ndim < a.ndim:
            return ((a - np.expand_dims(mns, axis=axis)) /
                    np.expand_dims(sstd, axis=axis))
        else:
            return (a - mns) / sstd, mns, sstd

    def augment(self,
                data,
                train_feats,
                train_labels,
                test_feats,
                test_labels,
                train_langs,
                test_langs,
                mns,
                sstd):
        data["train_feats"].append(train_feats)
        data["train_labels"].append(train_labels)
        data["test_feats"].append(test_feats)
        data["test_labels"].append(test_labels)
        data["train_langs"].append(train_langs)
        data["test_langs"].append(test_langs)
        data["train_labels_mns"].append(mns)
        data["train_labels_sstd"].append(sstd)


# get K fold data from features
class K_Fold_Spliter(Spliter):
    def __init__(self, org_data, standardize=False, k=5):
        Spliter.__init__(self, org_data, standardize)
        self.k = k

    def split(self):
        k_fold_data = {}
        models = list(self.org_data.keys())

        for i, model in enumerate(models):
            model_data = self.org_data[model]
            feats = model_data["feats"]
            labels = model_data["labels"]
            langs = model_data["langs"]

            lens = len(feats)
            assert lens == len(labels)

            logger.info(f"K fold splitter splitting {lens} experimental records for model {model} into {self.k} folds.")

            # main logic
            ex_per_fold = int(np.ceil(lens / self.k))
            for j in range(self.k):
                start = ex_per_fold * j
                end = ex_per_fold * (j + 1)
                if start < lens:
                    if j == 0:
                        k_fold_data[model] = deepcopy(self.block)

                    train_feats, train_labels, train_langs = \
                        pd.concat([feats.iloc[:start, :], feats.iloc[end:, :]], axis=0), \
                        pd.concat([labels.iloc[:start, :], labels.iloc[end:, :]], axis=0), \
                        pd.concat([langs.iloc[:start, :], langs.iloc[end:, :]], axis=0)

                    test_feats, test_labels = feats.iloc[start:end, :], labels.iloc[start:end, :]

                    test_langs = langs.iloc[start:end, :]
                    test_lang_count = len(test_langs)

                    logger.info(f"Fold {j} has {lens - test_lang_count} training experimental records "
                                f"and {test_lang_count} testing experimental records.")

                    # same as other spliter methods
                    self.run_assertion(train_feats, train_labels, test_feats, test_labels, test_lang_count)

                    train_feats, train_labels, test_feats, test_labels, mns, sstd = \
                        self.standardize_data(train_feats, train_labels, test_feats, test_labels)

                    # place all the k folds data
                    self.augment(k_fold_data[model], train_feats, train_labels, test_feats, test_labels,
                                 train_langs, test_langs, mns, sstd)

            actual_k = len(k_fold_data[model]["train_feats"])
            if actual_k < self.k:
                logger.info(
                    f"Actually split {lens} experimental records for model {model} into {actual_k < self.k} folds.")

        return k_fold_data


class MM_K_fold_Spliter(Spliter):
    def __init__(self, org_data, standardize=False, k=5):
        Spliter.__init__(self, org_data, standardize)
        self.k = k

    def split(self):
        k_fold_data = {"all": deepcopy(self.block)}

        model_data = self.org_data["all"]
        feats = model_data["feats"]
        labels = model_data["labels"]
        langs = model_data["langs"]

        d = defaultdict(list)
        for i, (index, lang) in enumerate(zip(langs.index, langs.values)):
            d[tuple(lang)].append(index)
        keys = list(d)

        df_lens = len(feats)
        lens = len(d)
        logger.info(f"MM K fold splitter splitting {lens} * model experimental records for model all into {self.k} folds.")

        # main logic
        ex_per_fold = int(np.ceil(lens / self.k))
        for j in range(self.k):
            start = ex_per_fold * j
            end = ex_per_fold * (j + 1)
            if start < lens:
                test_langs_key = keys[start: end]

                test_ids = []
                for lang in test_langs_key:
                    test_ids += d[lang]

                train_ids = list(langs.index)
                for test_id in test_ids:
                    train_ids.remove(test_id)

                train_feats, train_labels, train_langs = feats.loc[train_ids], labels.loc[train_ids], langs.loc[train_ids]
                test_feats, test_labels, test_langs = feats.loc[test_ids], labels.loc[test_ids], langs.loc[test_ids]

                test_lang_count = len(test_langs)
                logger.info(f"Fold {j} has {df_lens - test_lang_count} training experimental records "
                            f"and {test_lang_count} testing experimental records.")

                # same as other spliter methods
                self.run_assertion(train_feats, train_labels, test_feats, test_labels, test_lang_count)

                train_feats, train_labels, test_feats, test_labels, mns, sstd = \
                    self.standardize_data(train_feats, train_labels, test_feats, test_labels)

                # place all the k folds data
                self.augment(k_fold_data["all"], train_feats, train_labels, test_feats, test_labels,
                             train_langs, test_langs, mns, sstd)

        actual_k = len(k_fold_data["all"]["train_feats"])
        if actual_k < self.k:
            logger.info(
                f"Actually split {df_lens} experimental records into {actual_k < self.k} folds.")

        return k_fold_data


# random_split for data
class Random_Spliter(Spliter):
    def __init__(self, org_data, percentage=10, standardize=False):
        Spliter.__init__(self, org_data, standardize)
        self.percentage = percentage

    def split(self):
        data = {}
        models = list(self.org_data.keys())

        for i, model in enumerate(models):
            model_data = self.org_data[model]
            feats = model_data["feats"]
            labels = model_data["labels"]
            langs = model_data["langs"]
            assert len(feats) == len(labels)

            data[model] = deepcopy(self.block)

            lens = len(feats)
            test_lens = lens // self.percentage

            logger.info(f"Random splitter splitting {lens} experimental records for model {model} into "
                        f"{lens-test_lens} training experimental records and {test_lens} experimental records.")

            train_feats, train_labels = feats.iloc[test_lens:, :], labels.iloc[test_lens:, :]
            test_feats, test_labels = feats.iloc[:test_lens, :], labels.iloc[:test_lens, :]
            train_langs, test_langs = langs.iloc[test_lens:, :], langs.iloc[:test_lens, :]
            lang_count = len(test_langs)

            # same as other spliter methods
            self.run_assertion(train_feats, train_labels, test_feats, test_labels, lang_count)

            train_feats, train_labels, test_feats, test_labels, mns, sstd = \
                self.standardize_data(train_feats, train_labels, test_feats, test_labels)

            # place all the k folds data
            self.augment(data[model], train_feats, train_labels, test_feats, test_labels,
                         train_langs, test_langs, mns, sstd)
        return data


class Specific_Spliter(Spliter):
    def __init__(self, org_data, train_ids, test_ids, standardize=False):
        Spliter.__init__(self, org_data, standardize)
        self.train_ids = train_ids
        self.test_ids = test_ids

    def run_assertion(self, train_feats, train_labels, test_feats, test_labels, lang_count):
        assert type(train_feats) == pd.DataFrame, type(train_labels) == pd.DataFrame
        assert type(test_feats) == pd.DataFrame, type(test_labels) == pd.DataFrame
        assert len(test_feats) == len(test_labels) == lang_count

    def split(self):
        data = {}
        models = list(self.org_data.keys())

        for i, model in enumerate(models):
            data[model] = deepcopy(self.block)

            model_data = self.org_data[model]
            feats = model_data["feats"]
            labels = model_data["labels"]
            langs = model_data["langs"]
            assert len(feats) == len(labels)

            for train_id, test_id in zip(self.train_ids, self.test_ids):
                train_feats, train_labels, train_langs = feats.loc[train_id, :], labels.loc[train_id, :], \
                                                         langs.loc[train_id, :]
                test_feats, test_labels, test_langs = feats.loc[test_id, :], labels.loc[test_id, :], \
                                                      langs.loc[test_id, :]
                lens = len(feats)
                test_lang_count = len(test_langs)

                logger.info(f"Specific splitter splitting {lens} experimental records for model {model} into "
                            f"{lens - test_lang_count} training experimental records and {test_lang_count} test experimental records.")

                # same as other spliter methods
                self.run_assertion(train_feats, train_labels, test_feats, test_labels, test_lang_count)

                train_feats, train_labels, test_feats, test_labels, mns, sstd = \
                    self.standardize_data(train_feats, train_labels, test_feats, test_labels)

                # place all the k folds data
                self.augment(data[model], train_feats, train_labels, test_feats, test_labels,
                             train_langs, test_langs, mns, sstd)

        return data


# TODO: fix this
class Group_Spliter(Spliter):
    def __init__(self, org_data, group_columns, standardize=False):
        Spliter.__init__(self, org_data, standardize)
        self.group_columns = group_columns

    def split(self):
        data = {}
        for i, model in enumerate(self.org_data.keys()):
            model_data = self.org_data[model]
            feats = model_data["feats"]
            labels = model_data["labels"]
            langs = model_data["langs"]

            lens = len(feats)
            assert lens == len(labels)

            data[model] = deepcopy(self.block)
            pass


if __name__ == '__main__':
    data = read_data("ud", folder="/Users/mengzhouxia/dongdong/CMU/Neubig/nlppred", combine_models=False)
    k_fold_data = K_Fold_Spliter(data)
    k_fold_data.split()