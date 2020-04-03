from read_data import get_data_langs
from task_feats import get_mono
import pandas as pd
from itertools import combinations
from utils import standardize_feats_df
from read_data import augment, specific_split
from run_predictions import run_once_test
from scipy.special import comb
from utils import log_args
from logger import create_logger
import argparse
import numpy as np
import random
from train_model import calculate_rmse
from task_feats import task_eval_metrics
from sklearn.linear_model import LinearRegression

def init_logging():
    logger.info("Representativeness experiments running ...")
    logger.info("python3 " + " ".join(sys.argv))
    log_args(params)


def get_data_from_metric(data):
    return data["feats"], data["labels"], data["langs"], data["lang_pairs"]


def leave_one_out(l, id):
    return l[:id] + l[id+1:]


def flatten_datasets(task, shuffle=False):
    data = get_data_langs(task, shuffle=shuffle)
    mono = get_mono(task)
    metrics = list(data.keys())
    metrics = [metric for metric in metrics if metric != "langs" and metric != "lang_pairs"]
    feats = []
    labels = []
    lang_or_langpair = []
    org_data = {}
    for metric in metrics:
        if metric != "langs" and metric != "lang_pairs":
            feats_metric, labels_metric, langs_metric, lang_pairs_metric = get_data_from_metric(data[metric])
            feats.append(feats_metric)
            labels_metric.columns = ["metric"]
            labels.append(labels_metric)
            if mono:
                lang_or_langpair.append(langs_metric)
            else:
                lang_or_langpair.append(lang_pairs_metric)

    for i, feats_metric in enumerate(feats):
        for j, metric in enumerate(metrics):
            feats_metric["model_{}".format(j)] = 0
        feats_metric["model_{}".format(i)] = 1

    logger.info("Feats new: {}".format(", ".join(feats[0].columns)))

    for i, metric in enumerate(metrics):
        org_data[metric] = {"train_feats": pd.concat(leave_one_out(feats, i), axis=0),
                            "train_labels": pd.concat(leave_one_out(labels, i), axis=0),
                            "train_lang_or_lang_pair": pd.concat(leave_one_out(lang_or_langpair, i), axis=0),
                            "test_feats": feats[i],
                            "test_labels": labels[i],
                            "test_lang_or_lang_pair": lang_or_langpair[i]}
        # sanity check
        # print(metric)
        # test_labels = org_data[metric]["test_labels"]
        # train_labels = org_data[metric]["train_labels"]
        # test_labels = set(test_labels.iloc[:, 0].values.tolist())
        # train_labels = set(train_labels.iloc[:, 0].values.tolist())
        # print(test_labels - (test_labels - train_labels))
    return org_data

def initialize_re_block(metric, mono, *args):
    # Initialization
    re = {metric: {}}
    re[metric]["reg"] = {}
    re[metric]["train_rmse"] = {}
    re[metric]["test_rmse"] = {}
    re[metric]["test_preds"] = {}
    re[metric]["test_labels"] = {}
    re[metric]["test_lower_preds"] = {}
    re[metric]["test_upper_preds"] = {}
    re[metric]["test_langs"] = {}
    re[metric]["test_lang_pairs"] = {}
    if mono:
        re["ori_test_langs"] = args[0].sort_index()
    else:
        re["ori_test_lang_pairs"] = args[0].sort_index()
    return re

def run_ex(task, org_data, n=3, standardize=True, model="lr"):
    paras = {"mean_module": {"name": "constant_mean", "paras": {}}, "covar_module": {"name": "rbf", "paras": {}}}
    mono = get_mono(task)
    test_rmse = {}
    for metric in org_data:
        test_rmse[metric] = []
        data_metric = org_data[metric]
        ids = list(range(len(data_metric["test_feats"])))
        lens_ids = len(ids)
        options = combinations(ids, n)

        total_ex = int(comb(len(ids), n))
        rate = params.max_ex_per_metric / total_ex + 0.1

        logger.info("Running experiments with {} examples for a new model {}...".format(n, metric))
        logger.info("There are {} experiments running for model {}.. and we sample {} experimengts".format(total_ex, metric, params.max_ex_per_metric))
        finished_ex = 0
        for option in options:
            if random.random() > rate:
                continue
            data = {"train_feats": [], "train_labels": [], "test_feats": [], "test_labels": [],
                    "train_labels_mns": [], "train_labels_sstd": []}
            test_ids = list(set(ids) - set(option))
            option = list(option)
            train_feats = pd.concat([data_metric["train_feats"], data_metric["test_feats"].iloc[option]])
            train_labels = pd.concat([data_metric["train_labels"], data_metric["test_labels"].iloc[option]])
            train_lang_or_lang_pair = pd.concat([data_metric["train_lang_or_lang_pair"],
                                                data_metric["test_lang_or_lang_pair"].iloc[option]])
            test_feats = data_metric["test_feats"].iloc[test_ids]
            test_labels = data_metric["test_labels"].iloc[test_ids]
            test_lang_or_lang_pair = data_metric["test_lang_or_lang_pair"].iloc[option]
            data["test_lang_or_lang_pair"] = test_lang_or_lang_pair
            mns = None; sstd = None
            if standardize:
                train_feats, test_feats = standardize_feats_df(train_feats, test_feats)
                train_labels, test_labels, mns, sstd = standardize_feats_df(train_labels, test_labels, True)
            augment(data, train_feats, train_labels, test_feats, test_labels, mns, sstd)
            re = initialize_re_block(metric, mono, test_lang_or_lang_pair)
            re = run_once_test(data, 0, metric, re, model, get_rmse=True, get_ci=True, quantile=0.95,
                               standardize=standardize, paras=paras, verbose=False)
            test_rmse[metric].append(re[metric]['test_rmse'][0])

            finished_ex += 1
            if finished_ex % 100 == 0:
                logger.info("Progress: {}/{}, {:.2f}%, RMSE@{}: {:.2f}".format(
                    finished_ex, total_ex, finished_ex/total_ex*100, n, np.mean(test_rmse[metric])))
            if finished_ex == params.max_ex_per_metric:
                break
        logger.info("{} done! RMSE@{}: {:.2f}".format(metric, n, np.mean(test_rmse[metric])))
    logger.info("All experiements done! RMSE@{}: {:.2f}".format(n, np.mean(np.mean([test_rmse[metric] for metric in org_data]))))

def run_baseline(bl="mean"):
    if bl == "mean":
        data = get_data_langs(task, shuffle=False)
        metrics = list(data.keys())
        metrics = [metric for metric in metrics if metric != "langs" and metric != "lang_pairs"]
        labels = []
        for metric in metrics:
            feats_metric, labels_metric, langs_metric, lang_pairs_metric = get_data_from_metric(data[metric])
            labels.append(labels_metric)
        for i, metric in enumerate(metrics):
            train_labels = pd.concat(leave_one_out(labels, i), axis=1)
            test_labels = labels[i]
            pred = np.mean(train_labels.values, axis=1)
            print(pred.shape)
            print(test_labels.shape)
            rmse = calculate_rmse(pred, test_labels.values)
            logger.info("Mean baseline for metric {} is {:.2f} ..".format(metric, rmse))
    elif bl == "predictor":
        data = get_data_langs(task, shuffle=False, combine_models=True)
        metric_data = data["all"]
        feats = metric_data["feats"]; labels = metric_data["labels"]
        langs = metric_data["langs"]; lang_pairs = metric_data["lang_pairs"]
        for i in range(0, len(task_eval_metrics(task))):
            test_index = feats[feats["model_" + str(i)] == 1].index
            train_index = list(set(feats.index) - set(test_index))
            splited_data = specific_split(metric_data, train_index, test_index, loc=True)
            re = initialize_re_block("all", True, langs)
            run_once_test(splited_data, 0, "all", re, "xgboost", get_rmse=True, get_ci=False, quantile=0.95,
                          standardize=False, verbose=False)
            print(task_eval_metrics(task)[i], re["all"]['test_rmse'][0])


if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser(description="Models")
    parser.add_argument("--task", type=str, default="ud", help="the name of the running task")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--log", default="log", type=str, help="the log file")
    parser.add_argument("--n", type=int, default=0, help="How many should we add?")
    parser.add_argument("--max_ex_per_metric", type=int, default=10, help="Max experiments per metric")
    params = parser.parse_args()

    logger = create_logger(params.log, vb=params.verbose)
    init_logging()
    task = params.task

    # n = 5
    n = params.n
    run_ex(task=task, n=n, org_data=flatten_datasets(task), standardize=False, model="xgboost")

    # run_baseline("predictor")


