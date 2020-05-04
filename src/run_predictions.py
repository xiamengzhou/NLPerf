from train_model import train_regressor, test_regressor
from utils import convert_label, recover
from task_feats import task_att
from read_data import read_data, K_Fold_Spliter, Random_Spliter, Specific_Spliter, MM_K_fold_Spliter
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import torch
from logging import getLogger
from deprecated import deprecated

logger = getLogger()


# make sure that train_feats, train_labels, test_feats and test_labels are all data_frames
def run_once(train_feats,
             train_labels,
             test_feats,
             test_labels,
             mns,
             sstd,
             regressor="xgboost",
             get_ci=True,
             quantile=0.95,
             **kwarg):

    # convert label data frame to np array
    train_labels = convert_label(train_labels)
    test_labels = convert_label(test_labels)

    # training a regressor
    reg = train_regressor(train_feats=train_feats,
                          train_labels=train_labels,
                          regressor=regressor,
                          quantile=quantile,
                          verbose=False,
                          kwarg=kwarg)

    # get lower and upper bounds for boosting tree regressors
    lower_reg = None; upper_reg = None
    if get_ci:
        if isinstance(reg, xgb.XGBRegressor):
            lower_reg = train_regressor(train_feats, train_labels, regressor="lower_xgbq",
                                        quantile=quantile)
            upper_reg = train_regressor(train_feats, train_labels, regressor="upper_xgbq",
                                        quantile=quantile)
        elif isinstance(reg, GradientBoostingRegressor):
            lower_reg = train_regressor(train_feats, train_labels, regressor="lower_gb",
                                        quantile=quantile)
            upper_reg = train_regressor(train_feats, train_labels, regressor="upper_gb",
                                        quantile=quantile)

    train_preds, _, _, train_rmse, train_labels = test_regressor(reg, train_feats, train_labels, mns=mns, sstd=sstd)

    if len(test_feats) == 0:
        test_preds, test_lower_preds, test_upper_preds, test_rmse, test_labels = None, None, None, None, None
    else:
        test_preds, test_lower_preds, test_upper_preds, test_rmse, test_labels = test_regressor(reg, test_feats,
                                                                                               test_labels,
                                                                                               quantile=quantile,
                                                                                               lower_reg=lower_reg,
                                                                                               upper_reg=upper_reg,
                                                                                               mns=mns, sstd=sstd)

    return train_rmse, train_preds, test_rmse, test_preds, train_labels, test_labels, \
           test_upper_preds, test_lower_preds, reg


def augment_re(re, model, train_rmse, train_preds, test_rmse, test_preds, train_labels, test_labels,
               test_upper_preds, test_lower_preds, reg):
    re[model]["train_rmse"].append(train_rmse)
    re[model]["train_preds"].append(train_preds)
    re[model]["test_rmse"].append(test_rmse)
    re[model]["test_preds"].append(test_preds)
    re[model]["train_labels"].append(train_labels)
    re[model]["test_labels"].append(test_labels)
    re[model]["test_upper_preds"].append(test_upper_preds)
    re[model]["test_lower_preds"].append(test_lower_preds)
    re[model]["reg"].append(reg)

    # gpexact can't be serialized
    if type(reg) == tuple and isinstance(reg[0], torch.nn.Module):
        pass


def initialize_re_block(models):
    # Initialization
    re = {}
    for c in models:
        re[c] = {}
        re[c]["reg"] = []
        re[c]["train_rmse"] = []
        re[c]["test_rmse"] = []
        re[c]["train_preds"] = []
        re[c]["test_preds"] = []
        re[c]["train_labels"] = []
        re[c]["test_labels"] = []
        re[c]["test_lower_preds"] = []
        re[c]["test_upper_preds"] = []
    return re


def get_split_data(org_data, split_method="k_fold_split", **kwargs):
    if split_method == "random_split":
        splitter = Random_Spliter(org_data, percentage=kwargs["percentage"])
    elif split_method == "k_fold_split":
        splitter = K_Fold_Spliter(org_data, k=kwargs["k"])
    elif split_method == "specific_split":
        splitter = Specific_Spliter(org_data, kwargs["train_ids"], kwargs["test_ids"])
    elif split_method == "rep_k_fold_split":
        splitter = MM_K_fold_Spliter(org_data, k=kwargs["k"])
    else:
        return
    return splitter.split()


def get_baselines(split_data):
    baseline_re = {}
    for model in split_data:
        baseline_re[f"{model}"] = {}

    for model in split_data:
        model_data = split_data[model]
        baseline_re[f"{model}"] = {"test_preds": {}, "test_labels": []}
        test_preds = baseline_re[f"{model}"]["test_preds"]

        folds = len(model_data["train_feats"])

        # initialization for different types of baselines
        test_preds["k_split"] = []
        for lang_type in model_data["test_langs"][0].columns:
            test_preds[f"{lang_type}"] = []

        if len(split_data) > 1:
            test_preds["multi_model"] = []

        for i in range(folds):
            train_labels = model_data["train_labels"][i]
            test_labels = model_data["test_labels"][i]

            train_langs = model_data["train_langs"][i]
            test_langs = model_data["test_langs"][i]

            mns = model_data["train_labels_mns"][i]
            sstd = model_data["train_labels_sstd"][i]

            test_lens = len(test_labels)

            # k_split baseline
            k_split_test_preds = train_labels[model].mean()
            test_preds["k_split"].append(np.ones(test_lens) * k_split_test_preds)

            # lang wise baseline, source language, target language, transfer language
            lang_result_df = pd.DataFrame(columns=test_langs.columns, data=np.zeros(shape=(len(test_langs), len(test_langs.columns))))
            lang_result_df = lang_result_df.set_index(test_langs.index)
            for lang_type in test_langs.columns:
                for lang in test_langs[lang_type]:
                    lang_index = test_langs[test_langs[lang_type] == lang].index
                    langwise_mean = train_labels[train_langs[lang_type] == lang][model].mean()
                    # if no training data sharing the same language
                    if pd.isna(langwise_mean):
                        lang_result_df.loc[lang_index, lang_type] = k_split_test_preds
                    else:
                        lang_result_df.loc[lang_index, lang_type] = langwise_mean
                test_preds[lang_type].append(lang_result_df[lang_type].to_numpy())

            # model wise baseline
            if len(split_data) > 1:
                all_test_labels = [split_data[m]["test_labels"][i] for m in split_data if m != model]
                all_test_labels = pd.concat(all_test_labels, axis=1)
                multi_model_mean = all_test_labels.mean(axis=1).fillna(k_split_test_preds)
                test_preds["multi_model"].append(multi_model_mean.to_numpy())

            if mns is not None and sstd is not None:  # recover standardization
                for baseline_type in test_preds:
                    for i in range(test_preds[baseline_type]):
                        test_preds[baseline_type][i] = recover(mns, sstd, test_preds[baseline_type][i])
                test_labels = recover(mns, sstd, test_labels)

            baseline_re[f"{model}"]["test_labels"].append(test_labels)

    return baseline_re


def get_result(split_data,
               regressor="xgboost",
               get_ci=False,
               quantile=0.95,
               **kwargs):
    # Initialization
    re = initialize_re_block(list(split_data.keys()))

    # iterate through each fold
    for model in split_data:
        model_data = split_data[model]

        folds = len(model_data["train_feats"])

        for i in range(folds):
            train_feats = model_data["train_feats"][i]
            train_labels = model_data["train_labels"][i]

            test_feats = model_data["test_feats"][i]
            test_labels = model_data["test_labels"][i]

            mns = model_data["train_labels_mns"][i]
            sstd = model_data["train_labels_sstd"][i]

            train_rmse, train_preds, test_rmse, test_preds, train_labels_np, test_labels_np, \
                test_upper_preds, test_lower_preds, reg = \
                    run_once(train_feats, train_labels, test_feats, test_labels,
                             mns, sstd, regressor, get_ci, quantile, kwargs=kwargs)

            augment_re(re, model, train_rmse, train_preds, test_rmse, test_preds, train_labels_np, test_labels_np,
                       test_upper_preds, test_lower_preds, reg)
    return re


@deprecated
def sort_pred_refactor(re_dict, task, get_ci=False):
    # before sort {eval_metric: {"reg": [], "train_rmse": [], "test_preds": [], "test_rmse": [], "test_labels": [],
    #                            "test_lower_preds": [], "test_upper_preds": [],
    #                            "test_langs": [] or "test_lang_pairs": []}}
    # The reg is not necessary, because it won't be used anyway
    # The train_rmse and test_rmse can be used to calculate the mean manual
    # "reg", "test_preds", "test_labels", "test_lower_preds" and "test_upper_preds" can be poped from the dictionary
    # "metric_lower_preds", "metric_upper_preds", "metric_labels", "result_metric"
    # after sort {task: {eval_metric: {"reg": [], "train_rmse": [], "test_preds": [], "test_rmse": [],
    #                                  "test_labels": [], "test_lower_preds": [], "test_upper_preds": [],
    #                                  "test_langs": [] or "test_lang_pairs": [],
    #                                  "test_langs_sorted": [sorted] or "test_lang_pairs_sorted": [sorted],
    #                                  "result_metric": [sorted], "metric_labels": [sorted],
    #                                  "metric_lower_preds: [sorted], "metric_upper_preds": [sorted]}}}
    mono, multi_metric, _, _, _ = task_att(task)
    for eval_metric in re_dict[task]:
        print(re_dict[task].keys())
        if eval_metric != "ori_test_langs" and eval_metric != "ori_test_lang_pairs":
            test_preds = []
            reee = re_dict[task][eval_metric]
            k = len(re_dict[task][eval_metric]["test_langs"]) if mono else len(re_dict[task][eval_metric]["test_lang_pairs"])
            for i in range(k):
                test_pred = re_dict[task][eval_metric]["test_langs"][i] if mono \
                    else re_dict[task][eval_metric]["test_lang_pairs"][i]
                test_pred["preds"] = reee["test_preds"][i]
                test_pred["test_labels"] = reee["test_labels"][i]
                if get_ci:
                    test_pred["test_upper_preds"] = reee["test_upper_preds"][i]
                    test_pred["test_lower_preds"] = reee["test_lower_preds"][i]
                test_preds.append(test_pred)
            test_preds = pd.concat(test_preds)
            result = []
            labels = []
            if get_ci:
                lower_preds = []
                upper_preds = []
            if mono:
                langs = re_dict[task]["ori_test_langs"].values
                langs = langs.reshape(len(langs))
                for lang in langs:
                    se = test_preds[test_preds.iloc[:, 0] == lang]
                    if len(se) == 0:
                        result.append(np.nan)
                        labels.append(np.nan)
                        if get_ci:
                            lower_preds.append(np.nan)
                            upper_preds.append(np.nan)
                    else:
                        result.append(se["preds"].values[0])
                        labels.append(se["test_labels"].values[0])
                        if get_ci:
                            lower_preds.append(se["test_lower_preds"].values[0])
                            upper_preds.append(se["test_upper_preds"].values[0])
            else:
                for l1, l2 in re_dict[task]["ori_test_lang_pairs"].values:
                    se = test_preds[(test_preds.iloc[:, 0] == l1) & (test_preds.iloc[:, 1] == l2)]
                    if len(se) == 0:
                        result.append(np.nan)
                        labels.append(np.nan)
                        if get_ci:
                            lower_preds.append(np.nan)
                            upper_preds.append(np.nan)
                    else:
                        result.append(se["preds"].values[0])
                        labels.append(se["test_labels"].values[0])
                        if get_ci:
                            lower_preds.append(se["test_lower_preds"].values[0])
                            upper_preds.append(se["test_upper_preds"].values[0])

            re_dict[task][eval_metric]["result_{}".format(eval_metric)] = np.array(result)
            re_dict[task][eval_metric]["{}_labels".format(eval_metric)] = np.array(labels)

            if get_ci:
                re_dict[task][eval_metric]["{}_lower_preds".format(eval_metric)] = np.array(lower_preds)
                re_dict[task][eval_metric]["{}_upper_preds".format(eval_metric)] = np.array(upper_preds)


# currently only supports for tsf tasks
# bayesian optimization for finding the best transfer dataset
# settings -> parameters
@deprecated
def bayesian_optimization(task, k_fold_eval=False, regressor="xgboost",
                          get_rmse=True, get_ci=False, quantile=0.95, standardize=False):
    data, langs, lang_pairs = read_data(task, shuffle=k_fold_eval)
    re = {}

    for lang in langs:
        re[lang] = {"steps": 0, "langs": [], "ub": []}
        train_feats, train_labels, test_feats, test_labels, mns, sstd = (data, lang, lang_pairs)
        optimal_row = test_feats.iloc[[np.argmax(test_labels.values)]]
        optimal_lang = get_lang_from_feats(optimal_row)
        tsf_lang = -1
        print("The optimal transfer language for {} is {}.".format(lang, optimal_lang))
        while tsf_lang != optimal_lang: # should test with other stopping criterion
            reg = train_regressor(train_feats, train_labels, regressor=regressor)
            upper_reg, lower_reg = get_lower_upper_reg(reg, train_feats, train_labels, quantile)
            preds, lower_preds, upper_preds, rmse = \
                test_regressor(reg, test_feats, test_labels=None, get_rmse=get_rmse,
                               get_ci=get_ci, quantile=0.95, lower_reg=lower_reg, upper_reg=upper_reg,
                               mns=mns, sstd=sstd)
            upper_preds = recover(mns, sstd, upper_preds)
            ind = np.argmax(upper_preds)
            r_feats = test_feats.iloc[[ind]]
            r_labels = test_labels.iloc[[ind]]
            train_feats = pd.concat([train_feats, r_feats])
            train_labels = pd.concat([train_labels, r_labels])
            tsf_lang = get_lang_from_feats(r_feats)
            test_feats = test_feats.drop(test_feats.index[ind])
            test_labels = test_labels.drop(test_labels.index[ind])
            re[lang]["steps"] += 1
            re[lang]["langs"].append(tsf_lang)
            re[lang]["ub"].append(upper_preds[ind])
            print("Found {}!".format(tsf_lang))
    return re


@deprecated
def get_lower_upper_reg(reg, train_feats, train_labels, quantile):
    if isinstance(reg, xgb.XGBRegressor):
        lower_reg = train_regressor(train_feats, train_labels, regressor="lower_xgbq",
                                    quantile=quantile)
        upper_reg = train_regressor(train_feats, train_labels, regressor="upper_xgbq",
                                    quantile=quantile)
    elif isinstance(reg, GradientBoostingRegressor):
        lower_reg = train_regressor(train_feats, train_labels, regressor="lower_gb",
                                    quantile=quantile)
        upper_reg = train_regressor(train_feats, train_labels, regressor="upper_gb",
                                    quantile=quantile)
    else:
        raise KeyError
    return upper_reg, lower_reg


@deprecated
def get_lang_from_feats(row, ttt="tsf"):
    return [key[4:] for key in row.columns if key.startswith(ttt) and row[key].values[0] == 1.0][0]


