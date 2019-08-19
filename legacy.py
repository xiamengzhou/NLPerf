from xgboost import plot_importance
import xgboost as xgb
import numpy as np

def train_ranker():
    param = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'rank:ndcg'}
    param['nthread'] = 4
    param['eval_metric'] = 'ndcg'
    num_round = 10
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    return bst

def ranker_pred():
    ypred = bst.predict(dtest)
    ind = list(np.argsort(-ypred))
    plot_importance(bst)

# TODO: needs refatoring
def get_re(task, k_fold_eval=False, regressor="xgboost", get_rmse=True, get_ci=False, quantile=0.95):
    # when doing random k_fold_eval, we need to shuffle the data
    data, langs, lang_pairs = get_data_langs(task, shuffle=k_fold_eval)
    re = {"reg": {}, "train_rmse": {}, "test_rmse": {}, "test_preds": {}, "test_labels": {}, "test_lower_preds": {}, "test_upper_preds": {}}
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
        k_fold_data = get_k_fold_data(data, k, task)
        if task.startswith("tsf"):
            re["test_lang_pairs"] = {}
        elif task == "sf":
            re["test_langs"] = {}
            for c in k_fold_data["test_labels"][0].columns:
                re["reg"][c] = {}
                re["train_rmse"][c] = {}
                re["test_rmse"][c] = {}
                re["test_preds"][c] = {}
                re["test_labels"][c] = {}
                re["test_lower_preds"][c] = {}
                re["test_upper_preds"][c] = {}
        elif task == "bli":
            re["test_lang_pairs"] = {}
            for c in k_fold_data["test_labels"][0].columns:
                re["reg"][c] = {}
                re["train_rmse"][c] = {}
                re["test_rmse"][c] = {}
                re["test_preds"][c] = {}
                re["test_labels"][c] = {}
                re["test_lower_preds"][c] = {}
                re["test_upper_preds"][c] = {}
        elif task == "mi":
            re["test_lang_pairs"] = {}
        else:
            re["test_langs"] = {}
        for i in range(k):
            test_feats = k_fold_data["test_feats"][i]
            # in the case that there is less than 10 folds with limited data
            if len(test_feats) > 0:
                test_labels = k_fold_data["test_labels"][i]
                train_feats = k_fold_data["train_feats"][i]
                train_labels = k_fold_data["train_labels"][i]
                if task.startswith("tsf"):
                    test_lang_pairs = k_fold_data["test_lang_pairs"][i]
                    reg = train_regressor(train_feats, train_labels, regressor=regressor)
                    _, _, _, train_rmse = test_regressor(reg, train_feats, convert_label(train_labels))
                    lower_reg = None; upper_reg = None;
                    if get_ci:
                        if isinstance(reg, xgb.XGBRegressor):
                            lower_reg = train_regressor(train_feats, train_labels, regressor="lower_xgbq", quantile=quantile)
                            upper_reg = train_regressor(train_feats, train_labels, regressor="upper_xgbq", quantile=quantile)
                        elif isinstance(reg, GradientBoostingRegressor):
                            lower_reg = train_regressor(train_feats, train_labels, regressor="lower_gb", quantile=quantile)
                            upper_reg = train_regressor(train_feats, train_labels, regressor="upper_gb", quantile=quantile)
                    test_preds, test_lower_preds, test_upper_preds, test_rmse =                         test_regressor(reg, test_feats, convert_label(test_labels), get_rmse=get_rmse,                         get_ci=get_ci, quantile=quantile, lower_reg=lower_reg, upper_reg =upper_reg)
                    re["reg"][i] = reg
                    re["train_rmse"][i] = train_rmse
                    re["test_preds"][i] = test_preds
                    re["test_lang_pairs"][i] = test_lang_pairs
                    re["test_rmse"][i] = test_rmse
                    re["test_labels"][i] = convert_label(test_labels)
                    re["test_lower_preds"][i] = test_lower_preds
                    re["test_upper_preds"][i] = test_upper_preds
                elif task.startswith("bli"):
                    test_lang_pairs = k_fold_data["test_lang_pairs"][i]
                    re["test_lang_pairs"][i] = test_lang_pairs
                    # nasty (needs reorganizing)
                    train_tmp = pd.concat([train_feats, train_labels], axis=1)
                    test_tmp = pd.concat([test_feats, test_labels], axis=1)
                    eval_metric_len = len(train_labels.columns)
                    for eval_metric in train_labels.columns:
                        # remove nan values
                        train_tmp_ = train_tmp.dropna(subset=[eval_metric], axis=0)
                        train_feats = train_tmp_.iloc[:, :-3]
                        train_labels_ = convert_label(train_tmp_[eval_metric])
                        test_labels_ = convert_label(test_labels[eval_metric])
                        reg = train_regressor(train_feats, train_labels_, regressor=regressor)
                        lower_reg = None; upper_reg = None;
                        if get_ci:
                            if isinstance(reg, xgb.XGBRegressor):
                                lower_reg = train_regressor(train_feats, train_labels_, regressor="lower_xgbq", quantile=quantile)
                                upper_reg = train_regressor(train_feats, train_labels_, regressor="upper_xgbq", quantile=quantile)
                            elif isinstance(reg, GradientBoostingRegressor):
                                lower_reg = train_regressor(train_feats, train_labels_, regressor="lower_gb", quantile=quantile)
                                upper_reg = train_regressor(train_feats, train_labels_, regressor="upper_gb", quantile=quantile)
                        _, _, _, train_rmse = test_regressor(reg, train_feats, train_labels_)
                        test_preds, test_lower_preds, test_upper_preds, test_rmse =                             test_regressor(reg, test_feats, test_labels_, get_rmse=get_rmse,                             get_ci=get_ci, quantile=quantile, lower_reg=lower_reg, upper_reg =upper_reg)
                        re["reg"][eval_metric][i] = reg
                        re["train_rmse"][eval_metric][i] = train_rmse
                        re["test_preds"][eval_metric][i] = test_preds
                        re["test_rmse"][eval_metric][i] = test_rmse
                        re["test_labels"][eval_metric][i] = test_labels_
                        re["test_lower_preds"][eval_metric][i] = test_lower_preds
                        re["test_upper_preds"][eval_metric][i] = test_upper_preds
                elif task == "sf":
                    test_langs = k_fold_data["test_langs"][i] # Series
                    re["test_langs"][i] = pd.DataFrame(test_langs.values, columns=["test_langs"])
                    for eval_metric in train_labels.columns:
                        train_labels_ = convert_label(train_labels[eval_metric])
                        test_labels_ = convert_label(test_labels[eval_metric])
                        reg = train_regressor(train_feats, train_labels_, regressor=regressor)
                        lower_reg = None; upper_reg = None;
                        if get_ci:
                            if isinstance(reg, xgb.XGBRegressor):
                                lower_reg = train_regressor(train_feats, train_labels_, regressor="lower_xgbq", quantile=quantile)
                                upper_reg = train_regressor(train_feats, train_labels_, regressor="upper_xgbq", quantile=quantile)
                            elif isinstance(reg, GradientBoostingRegressor):
                                lower_reg = train_regressor(train_feats, train_labels_, regressor="lower_gb", quantile=quantile)
                                upper_reg = train_regressor(train_feats, train_labels_, regressor="upper_gb", quantile=quantile)
                        _, _, _, train_rmse = test_regressor(reg, train_feats, train_labels_)
                        test_preds, test_lower_preds, test_upper_preds, test_rmse =                             test_regressor(reg, test_feats, test_labels_, get_rmse=get_rmse,                             get_ci=get_ci, quantile=quantile, lower_reg=lower_reg, upper_reg =upper_reg)
                        re["reg"][eval_metric][i] = reg
                        re["train_rmse"][eval_metric][i] = train_rmse
                        re["test_preds"][eval_metric][i] = test_preds
                        re["test_rmse"][eval_metric][i] = test_rmse
                        re["test_labels"][eval_metric][i] = test_labels_
                        re["test_lower_preds"][eval_metric][i] = test_lower_preds
                        re["test_upper_preds"][eval_metric][i] = test_upper_preds
                elif task == "monomt":
                    test_langs = k_fold_data["test_langs"][i]
                    re["test_langs"][i] = pd.DataFrame(test_langs.values, columns=["test_langs"])
                    reg = train_regressor(train_feats, train_labels, regressor=regressor)
                    lower_reg = None; upper_reg = None;
                    if get_ci:
                        if isinstance(reg, xgb.XGBRegressor):
                            lower_reg = train_regressor(train_feats, train_labels, regressor="lower_xgbq", quantile=quantile)
                            upper_reg = train_regressor(train_feats, train_labels, regressor="upper_xgbq", quantile=quantile)
                        elif isinstance(reg, GradientBoostingRegressor):
                            lower_reg = train_regressor(train_feats, train_labels, regressor="lower_gb", quantile=quantile)
                            upper_reg = train_regressor(train_feats, train_labels, regressor="upper_gb", quantile=quantile)
                    _, _, _, train_rmse = test_regressor(reg, train_feats, convert_label(train_labels))
                    test_preds, test_lower_preds, test_upper_preds, test_rmse =                         test_regressor(reg, test_feats, convert_label(test_labels), get_rmse=get_rmse,                         get_ci=get_ci, quantile=quantile, lower_reg=lower_reg, upper_reg =upper_reg)
                    re["reg"][i] = reg
                    re["train_rmse"][i] = train_rmse
                    re["test_preds"][i] = test_preds
                    re["test_rmse"][i] = test_rmse
                    re["test_labels"][i] = convert_label(test_labels)
                    re["test_lower_preds"][i] = test_lower_preds
                    re["test_upper_preds"][i] = test_upper_preds
                elif task == "mi":
                    test_lang_pairs = k_fold_data["test_lang_pairs"][i]
                    re["test_lang_pairs"][i] = test_lang_pairs
                    reg = train_regressor(train_feats, train_labels, regressor=regressor)
                    lower_reg = None; upper_reg = None;
                    if get_ci:
                        if isinstance(reg, xgb.XGBRegressor):
                            lower_reg = train_regressor(train_feats, train_labels, regressor="lower_xgbq", quantile=quantile)
                            upper_reg = train_regressor(train_feats, train_labels, regressor="upper_xgbq", quantile=quantile)
                        elif isinstance(reg, GradientBoostingRegressor):
                            lower_reg = train_regressor(train_feats, train_labels, regressor="lower_gb", quantile=quantile)
                            upper_reg = train_regressor(train_feats, train_labels, regressor="upper_gb", quantile=quantile)
                    _, _, _, train_rmse = test_regressor(reg, train_feats, convert_label(train_labels))
                    test_preds, test_lower_preds, test_upper_preds, test_rmse =                         test_regressor(reg, test_feats, convert_label(test_labels), get_rmse=get_rmse,                         get_ci=get_ci, quantile=quantile, lower_reg=lower_reg, upper_reg =upper_reg)
                    re["reg"][i] = reg
                    re["train_rmse"][i] = train_rmse
                    re["test_preds"][i] = test_preds
                    re["test_rmse"][i] = test_rmse
                    re["test_labels"][i] = convert_label(test_labels)
                    re["test_lower_preds"][i] = test_lower_preds
                    re["test_upper_preds"][i] = test_upper_preds
                else:
                    break
        sort_pred({task: re}, task, langs, lang_pairs, k_fold_eval, get_ci=get_ci)
    return re

# TODO: needs refatoring
def sort_pred(re_dict, task, langs, lang_pairs, k_fold_eval=False, get_ci=False):
    if not k_fold_eval:
        for lang in langs:
            preds = re_dict[task]["test_preds"][lang]
            for p in preds:
                print(p)
    else:
        if task.startswith("tsf"):
            test_preds = []
            k = len(re_dict[task]["test_lang_pairs"])
            for i in range(k):
                test_lang_pairs = re_dict[task]["test_lang_pairs"][i]
                preds = re_dict[task]["test_preds"][i]
                test_lang_pairs["preds"] = preds
                test_lang_pairs["test_labels"] = re_dict[task]["test_labels"][i]
                if get_ci:
                    test_lang_pairs["test_upper_preds"] = re_dict[task]["test_upper_preds"][i]
                    test_lang_pairs["test_lower_preds"] = re_dict[task]["test_lower_preds"][i]
                test_preds.append(test_lang_pairs)
            test_preds = pd.concat(test_preds)
            result = []
            labels = []
            if get_ci:
                lower_preds = []
                upper_preds = []
            for l1, l2 in lang_pairs.values:
                se = test_preds[(test_preds.iloc[:, 0] == l1) & (test_preds.iloc[:, 1] == l2)]
                result.append(se["preds"].values[0])
                labels.append(se["test_labels"].values[0])
                if get_ci:
                    lower_preds.append(se["test_lower_preds"].values[0])
                    upper_preds.append(se["test_upper_preds"].values[0])
            re_dict[task]["result"] = np.array(result)
            re_dict[task]["labels"] = np.array(labels)
            if get_ci:
                re_dict[task]["result_lower_preds"] = np.array(lower_preds)
                re_dict[task]["result_upper_preds"] = np.array(upper_preds)
        elif task == "sf":
            k = len(re_dict[task]["test_langs"])
            test_preds_ = re_dict[task]["test_preds"]
            test_labels_ = re_dict[task]["test_labels"]
            test_lower_preds_ = re_dict[task]["test_lower_preds"]
            test_upper_preds_ = re_dict[task]["test_upper_preds"]
            test_preds = []
            for i in range(k):
                test_langs = re_dict[task]["test_langs"][i]
                for eval_metric in test_preds_:
                    test_langs[eval_metric] = test_preds_[eval_metric][i]
                    test_langs[eval_metric + "_labels"] = test_labels_[eval_metric][i]
                    if get_ci:
                        test_langs[eval_metric + "_lower_preds"] = test_lower_preds_[eval_metric][i]
                        test_langs[eval_metric + "_upper_preds"] = test_upper_preds_[eval_metric][i]
                test_preds.append(test_langs)
            test_preds = pd.concat(test_preds)
            for c in test_preds_:
                result = []
                labels = []
                if get_ci:
                    lower_preds = []
                    upper_preds = []
                for lang in langs:
                    se = test_preds[(test_preds.iloc[:, 0] == lang)]
                    result.append(se[c].values[0])
                    labels.append(se[c + "_labels"].values[0])
                    if get_ci:
                        lower_preds.append(se[c + "_lower_preds"].values[0])
                        upper_preds.append(se[c + "_upper_preds"].values[0])
                re_dict[task]["result_{}".format(c)] = np.array(result)
                re_dict[task]["{}_labels".format(c)] = np.array(labels)
                if get_ci:
                    re_dict[task]["{}_upper_preds".format(c)] = np.array(upper_preds)
                    re_dict[task]["{}_lower_preds".format(c)] = np.array(lower_preds)
        elif task == "monomt":
            k = len(re_dict[task]["test_langs"])
            test_preds = []
            for i in range(k):
                test_langs = re_dict[task]["test_langs"][i]
                preds = re_dict[task]["test_preds"][i]
                test_langs["preds"] = preds
                test_langs["test_labels"] = re_dict[task]["test_labels"][i]
                if get_ci:
                    test_langs["test_lower_preds"] = re_dict[task]["test_lower_preds"][i]
                    test_langs["test_upper_preds"] = re_dict[task]["test_upper_preds"][i]
                test_preds.append(test_langs)
            test_preds = pd.concat(test_preds)
            result = []
            labels = []
            if get_ci:
                lower_preds = []
                upper_preds = []
            for lang in langs:
                se = test_preds[(test_preds.iloc[:, 0] == lang)]
                result.append(se["preds"].values[0])
                labels.append(se["test_labels"].values[0])
                if get_ci:
                    lower_preds.append(se["test_lower_preds"].values[0])
                    upper_preds.append(se["test_upper_preds"].values[0])
            re_dict[task]["result"] = np.array(result)
            re_dict[task]["labels"] = np.array(labels)
            if get_ci:
                re_dict[task]["result_lower_preds"] = np.array(lower_preds)
                re_dict[task]["result_upper_preds"] = np.array(upper_preds)
        elif task == "bli":
            k = len(re_dict[task]["test_lang_pairs"])
            test_preds = []
            test_preds_ = re_dict[task]["test_preds"]
            test_labels_ = re_dict[task]["test_labels"]
            test_lower_preds_ = re_dict[task]["test_lower_preds"]
            test_upper_preds_ = re_dict[task]["test_upper_preds"]
            for i in range(k):
                test_lang_pairs = re_dict[task]["test_lang_pairs"][i]
                for eval_metric in test_preds_:
                    test_lang_pairs[eval_metric] = test_preds_[eval_metric][i]
                    test_lang_pairs[eval_metric + "_labels"] = test_labels_[eval_metric][i]
                    if get_ci:
                        test_lang_pairs[eval_metric + "_lower_preds"] = test_lower_preds_[eval_metric][i]
                        test_lang_pairs[eval_metric + "_upper_preds"] = test_upper_preds_[eval_metric][i]
                test_preds.append(test_lang_pairs)
            test_preds = pd.concat(test_preds)
            for c in test_preds_:
                result = []
                labels = []
                if get_ci:
                    lower_preds = []
                    upper_preds = []
                for l1, l2 in lang_pairs.values:
                    se = test_preds[(test_preds.iloc[:, 0] == l1) & (test_preds.iloc[:, 1] == l2)]
                    result.append(se[c].values[0])
                    labels.append(se[c + "_labels"].values[0])
                    if get_ci:
                        lower_preds.append(se[c + "_lower_preds"].values[0])
                        upper_preds.append(se[c + "_upper_preds"].values[0])
                re_dict[task]["result_{}".format(c)] = np.array(result)
                re_dict[task]["{}_labels".format(c)] = np.array(labels)
                if get_ci:
                    re_dict[task]["{}_lower_preds".format(c)] = np.array(lower_preds)
                    re_dict[task]["{}_upper_preds".format(c)] = np.array(upper_preds)
        elif task == "mi":
            k = len(re_dict[task]["test_lang_pairs"])
            test_preds = []
            for i in range(k):
                test_lang_pairs = re_dict[task]["test_lang_pairs"][i]
                preds = re_dict[task]["test_preds"][i]
                test_lang_pairs["preds"] = preds
                test_lang_pairs["test_labels"] = re_dict[task]["test_labels"][i]
                if get_ci:
                    test_lang_pairs["test_lower_preds"] = re_dict[task]["test_lower_preds"][i]
                    test_lang_pairs["test_upper_preds"] = re_dict[task]["test_upper_preds"][i]
                test_preds.append(test_lang_pairs)
            test_preds = pd.concat(test_preds)
            result = []
            labels = []
            if get_ci:
                lower_preds = []
                upper_preds = []
            for l1, l2 in lang_pairs.values:
                se = test_preds[(test_preds.iloc[:, 0] == l1) & (test_preds.iloc[:, 1] == l2)]
                result.append(se["preds"].values[0])
                labels.append(se["test_labels"].values[0])
                if get_ci:
                    lower_preds.append(se["test_lower_preds"].values[0])
                    upper_preds.append(se["test_upper_preds"].values[0])
            re_dict[task]["result"] = np.array(result)
            re_dict[task]["labels"] = np.array(labels)
            if get_ci:
                re_dict[task]["result_lower_preds"] = np.array(lower_preds)
                re_dict[task]["result_upper_preds"] = np.array(upper_preds)

# TODO: needs refatoring
def sort_pred(re_dict, task, langs, lang_pairs, k_fold_eval=False, get_ci=False):
    if not k_fold_eval:
        for lang in langs:
            preds = re_dict[task]["test_preds"][lang]
            for p in preds:
                print(p)
    else:
        if task.startswith("tsf"):
            test_preds = []
            k = len(re_dict[task]["test_lang_pairs"])
            for i in range(k):
                test_lang_pairs = re_dict[task]["test_lang_pairs"][i]
                preds = re_dict[task]["test_preds"][i]
                test_lang_pairs["preds"] = preds
                test_lang_pairs["test_labels"] = re_dict[task]["test_labels"][i]
                if get_ci:
                    test_lang_pairs["test_upper_preds"] = re_dict[task]["test_upper_preds"][i]
                    test_lang_pairs["test_lower_preds"] = re_dict[task]["test_lower_preds"][i]
                test_preds.append(test_lang_pairs)
            test_preds = pd.concat(test_preds)
            result = []
            labels = []
            if get_ci:
                lower_preds = []
                upper_preds = []
            for l1, l2 in lang_pairs.values:
                se = test_preds[(test_preds.iloc[:, 0] == l1) & (test_preds.iloc[:, 1] == l2)]
                result.append(se["preds"].values[0])
                labels.append(se["test_labels"].values[0])
                if get_ci:
                    lower_preds.append(se["test_lower_preds"].values[0])
                    upper_preds.append(se["test_upper_preds"].values[0])
            re_dict[task]["result"] = np.array(result)
            re_dict[task]["labels"] = np.array(labels)
            if get_ci:
                re_dict[task]["result_lower_preds"] = np.array(lower_preds)
                re_dict[task]["result_upper_preds"] = np.array(upper_preds)
        elif task == "sf":
            k = len(re_dict[task]["test_langs"])
            test_preds_ = re_dict[task]["test_preds"]
            test_labels_ = re_dict[task]["test_labels"]
            test_lower_preds_ = re_dict[task]["test_lower_preds"]
            test_upper_preds_ = re_dict[task]["test_upper_preds"]
            test_preds = []
            for i in range(k):
                test_langs = re_dict[task]["test_langs"][i]
                for eval_metric in test_preds_:
                    test_langs[eval_metric] = test_preds_[eval_metric][i]
                    test_langs[eval_metric + "_labels"] = test_labels_[eval_metric][i]
                    if get_ci:
                        test_langs[eval_metric + "_lower_preds"] = test_lower_preds_[eval_metric][i]
                        test_langs[eval_metric + "_upper_preds"] = test_upper_preds_[eval_metric][i]
                test_preds.append(test_langs)
            test_preds = pd.concat(test_preds)
            for c in test_preds_:
                result = []
                labels = []
                if get_ci:
                    lower_preds = []
                    upper_preds = []
                for lang in langs:
                    se = test_preds[(test_preds.iloc[:, 0] == lang)]
                    result.append(se[c].values[0])
                    labels.append(se[c + "_labels"].values[0])
                    if get_ci:
                        lower_preds.append(se[c + "_lower_preds"].values[0])
                        upper_preds.append(se[c + "_upper_preds"].values[0])
                re_dict[task]["result_{}".format(c)] = np.array(result)
                re_dict[task]["{}_labels".format(c)] = np.array(labels)
                if get_ci:
                    re_dict[task]["{}_upper_preds".format(c)] = np.array(upper_preds)
                    re_dict[task]["{}_lower_preds".format(c)] = np.array(lower_preds)
        elif task == "monomt":
            k = len(re_dict[task]["test_langs"])
            test_preds = []
            for i in range(k):
                test_langs = re_dict[task]["test_langs"][i]
                preds = re_dict[task]["test_preds"][i]
                test_langs["preds"] = preds
                test_langs["test_labels"] = re_dict[task]["test_labels"][i]
                if get_ci:
                    test_langs["test_lower_preds"] = re_dict[task]["test_lower_preds"][i]
                    test_langs["test_upper_preds"] = re_dict[task]["test_upper_preds"][i]
                test_preds.append(test_langs)
            test_preds = pd.concat(test_preds)
            result = []
            labels = []
            if get_ci:
                lower_preds = []
                upper_preds = []
            for lang in langs:
                se = test_preds[(test_preds.iloc[:, 0] == lang)]
                result.append(se["preds"].values[0])
                labels.append(se["test_labels"].values[0])
                if get_ci:
                    lower_preds.append(se["test_lower_preds"].values[0])
                    upper_preds.append(se["test_upper_preds"].values[0])
            re_dict[task]["result"] = np.array(result)
            re_dict[task]["labels"] = np.array(labels)
            if get_ci:
                re_dict[task]["result_lower_preds"] = np.array(lower_preds)
                re_dict[task]["result_upper_preds"] = np.array(upper_preds)
        elif task == "bli":
            k = len(re_dict[task]["test_lang_pairs"])
            test_preds = []
            test_preds_ = re_dict[task]["test_preds"]
            test_labels_ = re_dict[task]["test_labels"]
            test_lower_preds_ = re_dict[task]["test_lower_preds"]
            test_upper_preds_ = re_dict[task]["test_upper_preds"]
            for i in range(k):
                test_lang_pairs = re_dict[task]["test_lang_pairs"][i]
                for eval_metric in test_preds_:
                    test_lang_pairs[eval_metric] = test_preds_[eval_metric][i]
                    test_lang_pairs[eval_metric + "_labels"] = test_labels_[eval_metric][i]
                    if get_ci:
                        test_lang_pairs[eval_metric + "_lower_preds"] = test_lower_preds_[eval_metric][i]
                        test_lang_pairs[eval_metric + "_upper_preds"] = test_upper_preds_[eval_metric][i]
                test_preds.append(test_lang_pairs)
            test_preds = pd.concat(test_preds)
            for c in test_preds_:
                result = []
                labels = []
                if get_ci:
                    lower_preds = []
                    upper_preds = []
                for l1, l2 in lang_pairs.values:
                    se = test_preds[(test_preds.iloc[:, 0] == l1) & (test_preds.iloc[:, 1] == l2)]
                    result.append(se[c].values[0])
                    labels.append(se[c + "_labels"].values[0])
                    if get_ci:
                        lower_preds.append(se[c + "_lower_preds"].values[0])
                        upper_preds.append(se[c + "_upper_preds"].values[0])
                re_dict[task]["result_{}".format(c)] = np.array(result)
                re_dict[task]["{}_labels".format(c)] = np.array(labels)
                if get_ci:
                    re_dict[task]["{}_lower_preds".format(c)] = np.array(lower_preds)
                    re_dict[task]["{}_upper_preds".format(c)] = np.array(upper_preds)
        elif task == "mi":
            k = len(re_dict[task]["test_lang_pairs"])
            test_preds = []
            for i in range(k):
                test_lang_pairs = re_dict[task]["test_lang_pairs"][i]
                preds = re_dict[task]["test_preds"][i]
                test_lang_pairs["preds"] = preds
                test_lang_pairs["test_labels"] = re_dict[task]["test_labels"][i]
                if get_ci:
                    test_lang_pairs["test_lower_preds"] = re_dict[task]["test_lower_preds"][i]
                    test_lang_pairs["test_upper_preds"] = re_dict[task]["test_upper_preds"][i]
                test_preds.append(test_lang_pairs)
            test_preds = pd.concat(test_preds)
            result = []
            labels = []
            if get_ci:
                lower_preds = []
                upper_preds = []
            for l1, l2 in lang_pairs.values:
                se = test_preds[(test_preds.iloc[:, 0] == l1) & (test_preds.iloc[:, 1] == l2)]
                result.append(se["preds"].values[0])
                labels.append(se["test_labels"].values[0])
                if get_ci:
                    lower_preds.append(se["test_lower_preds"].values[0])
                    upper_preds.append(se["test_upper_preds"].values[0])
            re_dict[task]["result"] = np.array(result)
            re_dict[task]["labels"] = np.array(labels)
            if get_ci:
                re_dict[task]["result_lower_preds"] = np.array(lower_preds)
                re_dict[task]["result_upper_preds"] = np.array(upper_preds)
