from train_model import train_regressor, test_regressor
from utils import convert_label
from task_feats import task_att
from read_data import get_data_langs, get_k_fold_data, get_transfer_data_by_group
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np

def get_re_refactor(task, k_fold_eval=False, regressor="xgboost", get_rmse=True, get_ci=False, quantile=0.95):
    # when doing random k_fold_eval, we need to shuffle the data
    mono, multi_metric, _ = task_att(task)
    feats, labels, langs, lang_pairs = get_data_langs(task, shuffle=k_fold_eval)
    re = {}
    if not k_fold_eval:
        for n, lang in enumerate(langs):
            train_feats, train_labels, test_feats, test_labels = get_transfer_data_by_group(data, lang, lang_pairs)
            reg = train_regressor(train_feats, train_labels)
            _, train_rmse = test_regressor(reg, train_feats, convert_label(train_labels))
            test_preds, test_rmse = test_regressor(reg, test_feats, convert_label(test_labels))
            re["reg"][lang] = reg
            re["train_rmse"][lang] = train_rmse
            re["test_preds"][lang] = test_preds
            re["test_rmse"][lang] = test_rmse
    else:
        k = 10
        k_fold_data = get_k_fold_data(feats, labels, lang_pairs, langs, k, task)
        # Initialization
        if mono:
            re["test_langs"] = {}
        else:
            re["test_lang_pairs"] = {}
        for c in k_fold_data:
            re[c] = {}
            re[c]["reg"] = {}
            re[c]["train_rmse"] = {}
            re[c]["test_rmse"] = {}
            re[c]["test_preds"] = {}
            re[c]["test_labels"] = {}
            re[c]["test_lower_preds"] = {}
            re[c]["test_upper_preds"] = {}

        # iterate through each fold
        for eval_metric in k_fold_data:
            data = k_fold_data[eval_metric]
            for i in range(k):
                test_feats = data["test_feats"][i]
                # in the case that there is less than 10 folds with limited data
                if len(test_feats) > 0:
                    # print(len(test_feats), len(data["test_labels"][i]), len(data["train_feats"][i]),
                    #       len(data["train_labels"][i]))
                    test_labels = data["test_labels"][i]
                    train_feats = data["train_feats"][i]
                    train_labels = data["train_labels"][i]
                    if mono:
                        test_langs = data["test_langs"][i]  # Series
                        re["test_langs"][i] = pd.DataFrame(test_langs.values, columns=["test_langs"])
                    else:
                        test_lang_pairs = data["test_lang_pairs"][i]
                        re["test_lang_pairs"][i] = test_lang_pairs

                    train_tmp = pd.concat([train_feats, train_labels], axis=1)
                    train_tmp_ = train_tmp.dropna(subset=[eval_metric], axis=0)
                    train_feats = train_tmp_.iloc[:, :-1]
                    train_labels_ = convert_label(train_tmp_[eval_metric])
                    test_labels_ = convert_label(test_labels)
                    reg = train_regressor(train_feats, train_labels_, regressor=regressor)
                    lower_reg = None; upper_reg = None
                    if get_ci:
                        if isinstance(reg, xgb.XGBRegressor):
                            lower_reg = train_regressor(train_feats, train_labels_, regressor="lower_xgbq",
                                                        quantile=quantile)
                            upper_reg = train_regressor(train_feats, train_labels_, regressor="upper_xgbq",
                                                        quantile=quantile)
                        elif isinstance(reg, GradientBoostingRegressor):
                            lower_reg = train_regressor(train_feats, train_labels_, regressor="lower_gb",
                                                        quantile=quantile)
                            upper_reg = train_regressor(train_feats, train_labels_, regressor="upper_gb",
                                                        quantile=quantile)
                    _, _, _, train_rmse = test_regressor(reg, train_feats, train_labels_)
                    test_preds, test_lower_preds, test_upper_preds, test_rmse = test_regressor(reg, test_feats,
                                                                                               test_labels_,
                                                                                               get_rmse=get_rmse,
                                                                                               get_ci=get_ci,
                                                                                               quantile=quantile,
                                                                                               lower_reg=lower_reg,
                                                                                               upper_reg=upper_reg)
                    re[eval_metric]["reg"][i] = reg
                    re[eval_metric]["train_rmse"][i] = train_rmse
                    re[eval_metric]["test_preds"][i] = test_preds
                    re[eval_metric]["test_rmse"][i] = test_rmse
                    re[eval_metric]["test_labels"][i] = test_labels_
                    if get_ci:
                        re[eval_metric]["test_lower_preds"][i] = test_lower_preds
                        re[eval_metric]["test_upper_preds"][i] = test_upper_preds
        # before sort {eval_metric: {"reg": [], "train_rmse": [], "test_preds": [], "test_rmse": [], "test_labels": [],
        #                            "test_lower_preds": [], "test_upper_preds": []}}
        sort_pred_refactor({task: re}, task, langs, lang_pairs, k_fold_eval, get_ci=get_ci)
    return re

def sort_pred_refactor(re_dict, task, langs, lang_pairs, k_fold_eval=False, get_ci=False):
    # before sort {eval_metric: {"reg": [], "train_rmse": [], "test_preds": [], "test_rmse": [], "test_labels": [],
    #                            "test_lower_preds": [], "test_upper_preds": []},
    #              "test_langs": [] or "test_lang_pairs": []}
    # The reg is not necessary, because it won't be used anyway
    # The train_rmse and test_rmse can be used to calculate the mean manual
    # "reg", "test_preds", "test_labels", "test_lower_preds" and "test_upper_preds" can be poped from the dictionary
    # after sort {task: {eval_metric: {"reg": [], "train_rmse": [], "test_preds": [], "test_rmse": [],
    #                                  "test_labels": [], "test_lower_preds": [], "test_upper_preds": [],
    #                                  "result_metric": [sorted], "metric_labels": [sorted],
    #                                  "metric_lower_preds: [sorted], "metric_upper_preds": [sorted]}}}
    mono, multi_metric, _ = task_att(task)
    if not k_fold_eval:
        for lang in langs:
            preds = re_dict[task]["test_preds"][lang]
            for p in preds:
                print(p)
    else:
        for eval_metric in re_dict[task]:
            if eval_metric != "test_langs" and eval_metric != "test_lang_pairs":
                test_preds = []
                reee = re_dict[task][eval_metric]
                k = len(re_dict[task]["test_langs"]) if mono else len(re_dict[task]["test_lang_pairs"])
                for i in range(k):
                    test_pred = re_dict[task]["test_langs"][i] if mono else re_dict[task]["test_lang_pairs"][i]
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
                    for lang in langs:
                        se = test_preds[(test_preds.iloc[:, 0] == lang)]
                        result.append(se["preds"].values[0])
                        labels.append(se["test_labels"].values[0])
                        if get_ci:
                            lower_preds.append(se["test_lower_preds"].values[0])
                            upper_preds.append(se["test_upper_preds"].values[0])
                else:
                    for l1, l2 in lang_pairs.values:
                        se = test_preds[(test_preds.iloc[:, 0] == l1) & (test_preds.iloc[:, 1] == l2)]
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
def bayesian_optimization(task, k_fold_eval=False, regressor="xgboost", get_rmse=True, get_ci=False, quantile=0.95):
    data, langs, lang_pairs = get_data_langs(task, shuffle=k_fold_eval)
    re = {}
    for lang in langs:
        re[lang] = {"steps": 0, "langs": [], "ub": []}
        train_feats, train_labels, test_feats, test_labels = get_transfer_data_by_group(data, lang, lang_pairs)
        optimal_row = test_feats.iloc[[np.argmax(test_labels.values)]]
        optimal_lang = get_lang_from_feats(optimal_row)
        tsf_lang = -1
        print("The optimal transfer language for {} is {}.".format(lang, optimal_lang))
        while tsf_lang != optimal_lang: # should test with other stopping criterion
            reg = train_regressor(train_feats, train_labels, regressor=regressor)
            upper_reg, lower_reg = get_lower_upper_reg(reg, train_feats, train_labels, quantile)
            preds, lower_preds, upper_preds, rmse = \
                test_regressor(reg, test_feats, test_labels=None, get_rmse=get_rmse,
                               get_ci=get_ci, quantile=0.95, lower_reg=lower_reg, upper_reg=upper_reg)
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


def get_lang_from_feats(row, ttt="tsf"):
    return [key[4:] for key in row.columns if key.startswith(ttt) and row[key].values[0] == 1.0][0]


