import numpy as np
from xgboost import plot_importance
import math

from src.read_data import read_data
from src.run_predictions import get_result, get_split_data, get_baselines
from src.task_feats import task_eval_columns, get_tasks
from src.train_model import calculate_rmse
from src.logger import create_logger
from deprecated import deprecated

@deprecated
def aget_result(regressor="xgboost", tasks="all", split_method="k_split", get_rmse=True,
               get_ci=False, quantile=0.95, standardize=False, paras=None, selected_feats=None, src_tgt="src",
               combine_model=False, shuffle=True):
    re_dict = {}
    if tasks == "all":
        tasks = ["monomt", "sf", "bli", "mi", "tsfel", "tsfmt", "tsfpos", "tsfparsing", "wiki", "ud"]
    for task in tasks:
        org_data = read_data(task, shuffle=shuffle, selected_feats=selected_feats, combine_models=combine_model)
        re = get_result(org_data=org_data, task=task, regressor=regressor, split_method=split_method,
                             get_rmse=get_rmse, get_ci=get_ci, quantile=quantile, standardize=standardize,
                             paras=paras, src_tgt=src_tgt)
        re_dict[task] = re
        print("{} is done!".format(task))
    return re_dict

@deprecated
def plot(ree, type="weight", lang="ara", task="mt"):
    task_re = ree[task]
    print(task_re["rmse"][lang])
    ax = plot_importance(task_re["reg"][lang], importance_type=type)
    fig = ax.figure
    fig.set_size_inches(5, 30)

@deprecated
def get_metric_deprecated(re, metric="test_rmse"):
    if metric == "mean_test_rmse":
        print("TSF_MT", np.mean(list(re["tsfmt"]["test_rmse"].values())))
        print("TSF_EL", np.mean(list(re["tsfel"]["test_rmse"].values())))
        print("TSF_POS", np.mean(list(re["tsfpos"]["test_rmse"].values())))
        print("TSF_PARSING", np.mean(list(re["tsfparsing"]["test_rmse"].values())))
        print("MONO_MT", np.mean(list(re["monomt"]["test_rmse"].values())))
        print("MI", np.mean(list(re["mi"]["test_rmse"].values())))
        print("SF_Keywords_F1", np.mean(list(re["sf"]["test_rmse"]["Keywords_F1"].values())))
        print("SF_Keywords_Precision", np.mean(list(re["sf"]["test_rmse"]["Keywords_Precision"].values())))
        print("SF_Keywords_Recall", np.mean(list(re["sf"]["test_rmse"]["Keywords_Recall"].values())))
        print("SF_NN_F1", np.mean(list(re["sf"]["test_rmse"]["NN_F1"].values())))
        print("SF_NN_Precision", np.mean(list(re["sf"]["test_rmse"]["NN_Precision"].values())))
        print("SF_NN_Recall", np.mean(list(re["sf"]["test_rmse"]["NN_Recall"].values())))
        print("BLI_MUSE", np.mean(list(re["bli"]["test_rmse"]["MUSE"].values())))
        print("BLI_Artetxe17", np.mean(list(re["bli"]["test_rmse"]["Artetxe17"].values())))
        print("BLI_Artetxe16", np.mean(list(re["bli"]["test_rmse"]["Artetxe16"].values())))
    elif metric == "test_rmse":
        for task in re:
            keys = list(re[task].keys())
            for key in keys:
                if key.startswith("result"):
                    if key != "result" and key != "result_upper_preds" and key != "result_lower_preds":
                        print("{}_{}".format(task.capitalize(), key[7:].capitalize()), calculate_rmse(re[task][key], re[task]["{}_labels".format(key[7:])]))
                    elif key == "result":
                        print("{}".format(task.capitalize()), calculate_rmse(re[task][key], re[task]["labels"]))
            
@deprecated
def get_metric_refactor(re, metric="test_rmse"):
    if metric == "test_rmse":
        for task in re:
            for eval_metric in re[task].keys():
                if eval_metric != "test_langs" or eval_metric != "test_lang_pairs":
                    reee = re[task][eval_metric]
                    print("{}_{}".format(task.capitalize(), eval_metric.capitalize()),
                          calculate_rmse(re[task][eval_metric], re[task]["{}_labels".format(eval_metric[7:])]))

@deprecated
def get_baseline(tasks=None):
    if tasks is None:
        tasks = get_tasks()
    for task in tasks:
        org_data = read_data(task, shuffle=False)
        metrics = task_eval_columns(task)
        rmses = []
        for metric in metrics:
            labels = org_data[metric]["labels"].values
            preds = np.mean(labels).repeat(len(labels))
            rmse = calculate_rmse(preds, labels)
            rmses.append(rmse)
            print("Mean baseline for task {} and metric {} is rmse {:.2f}".format(task, metric, rmse))
        print(f"Mean: {np.mean(rmses)}")

@deprecated
def get_model_baseline(tasks=None):
    from copy import deepcopy
    if tasks is None:
        tasks = get_tasks()
    for task in tasks:
        org_data = read_data(task, shuffle=False)
        metrics = task_eval_columns(task)
        rmses = []
        for metric in metrics:
            others = deepcopy(metrics)
            others.remove(metric)
            labelss = []
            for other in others:
                print(other)
                labels = org_data[other]["labels"].values
                labelss.append(labels)
            labels = org_data[metric]["labels"].values
            preds = sum(labelss) / len(labelss)
            rmse = calculate_rmse(preds, labels)
            rmses.append(rmse)
            print("model mean baseline for task {} and metric {} is rmse {:.2f}".format(task, metric, rmse))
        print(f"Mean: {np.mean(rmses)}")


def log_results_for_one_run(split_data, re, logger):
    for model in re:
        for i, test_rmse in enumerate(re[model]["test_rmse"]):
            logger.info(f"Model: {model}, Fold: {i}, "
                        f"Training data: {len(split_data[model]['train_feats'][i])} "
                        f"Test data: {len(split_data[model]['test_feats'][i])} "
                        f"Test rmse: {test_rmse}")
        logger.info(f"Model: {model}, Overall test rmse: {re[model]['test_rmse_all']}")


def log_results_for_baseline(re, logger):
    for model in re:
        for type in re[model]["rmse"]:
            rmse = re[model]["rmse"][type]
            logger.info(f"Model: {model}, Baseline: {type}, Overall test rmse: {rmse} ")


def aggregate_k_split_result(re):
    for model in re:
        test_preds = []
        test_labels = []
        for test_pred, test_label in zip(re[model]["test_preds"], re[model]["test_labels"]):
            test_preds.append(test_pred)
            test_labels.append(test_label)
        test_preds = np.concatenate(test_preds)
        test_labels = np.concatenate(test_labels)
        test_rmse = calculate_rmse(test_preds, test_labels)
        re[model]["test_rmse_all"] = test_rmse


def aggregate_k_split_baseline_result(re):
    for model in re:
        re[model]["rmse"] = {}
        for baseline_type in re[model]["test_preds"]:
            test_preds = []
            test_labels = []
            for test_pred, test_label in zip(re[model]["test_preds"][baseline_type], re[model]["test_labels"]):
                test_preds.append(test_pred)
                test_labels.append(test_label)
            test_preds = np.concatenate(test_preds)
            test_labels = np.concatenate(test_labels)
            test_rmse = calculate_rmse(test_preds, test_labels)
            re[model]["rmse"][baseline_type] = test_rmse


def specific_evaluation(task,
                        regressor,
                        get_ci=False,
                        quantile=0.95):
    logger = create_logger(f"logs/specific_evaluate_{task}")

    logger.info("*" * 50)
    logger.info(f"Running specific evaluation...")

    org_data = read_data(task,
                         shuffle=False,
                         folder="/Users/mengzhouxia/dongdong/CMU/Neubig/nlppred",
                         combine_models=False)

    k = 5
    train_ids = []; test_ids = []
    feats = list(org_data.values())[0]["feats"]
    ex_per_fold = math.ceil(len(feats) / 5)

    index = feats.index
    for i in range(k):
        test_id = feats.index[i * ex_per_fold: (i + 1) * ex_per_fold]
        train_id = index.delete(test_id)
        train_ids.append(train_id)
        test_ids.append(test_id)

    split_data = get_split_data(org_data, "specific_split", train_ids=train_ids, test_ids=test_ids)
    re = get_result(split_data, regressor, get_ci, quantile)
    aggregate_k_split_result(re)
    log_results_for_one_run(split_data, re, logger)

    baseline_re = get_baselines(split_data)
    aggregate_k_split_baseline_result(baseline_re)
    log_results_for_baseline(baseline_re, logger)


def k_fold_evaluation(task,
                      shuffle,
                      selected_feats,
                      combine_models,
                      regressor,
                      get_ci=False,
                      quantile=0.95,
                      k=5,
                      num_running=5):
    logger = create_logger(f"logs/{task}_{k}fold.log")

    test_rmse_all = {}
    test_rmse_all_baseline = {}

    for i in range(num_running):
        logger.info("*"*50)
        logger.info(f"Running k fold evaulation [{i+1}/{num_running}]...")
        org_data = read_data(task,
                             shuffle=shuffle,
                             folder="/Users/mengzhouxia/dongdong/CMU/Neubig/nlppred",
                             selected_feats=selected_feats,
                             combine_models=combine_models)

        split_data = get_split_data(org_data, "k_fold_split", k=k)
        re = get_result(split_data, regressor, get_ci, quantile)
        aggregate_k_split_result(re)
        log_results_for_one_run(split_data, re, logger)

        baseline_re = get_baselines(split_data)
        aggregate_k_split_baseline_result(baseline_re)
        log_results_for_baseline(baseline_re, logger)

        for model in split_data:
            test_rmse_all_baseline[model] = {}
            for type in baseline_re[model]["rmse"]:
                test_rmse_all_baseline[model][type] = []
            if i == 0:
                pass
                test_rmse_all[model] = [re[model]["test_rmse_all"]]
            else:
                test_rmse_all[model].append(re[model]["test_rmse_all"])
                pass
            for type in baseline_re[model]["rmse"]:
                test_rmse_all_baseline[model][type].append(baseline_re[model]["rmse"][type])

    logger.info("*"*50)

    avg_rmse_for_all_models = {"standard": []}
    avg_rmse_for_all_models.update({type: [] for type in test_rmse_all_baseline[list(test_rmse_all.keys())[0]]})

    for model in test_rmse_all:
        lens = len(test_rmse_all[model])
        avg_test_rmse_all = sum(test_rmse_all[model]) / lens
        avg_rmse_for_all_models["standard"].append(avg_test_rmse_all)
        logger.info(f"Model {model}: The average rmse for {num_running} runs is {avg_test_rmse_all}")

        for type in test_rmse_all_baseline[model]:
            lens = len(test_rmse_all_baseline[model][type])
            avg_test_rmse_all = sum(test_rmse_all_baseline[model][type]) / lens
            avg_rmse_for_all_models[type].append(avg_test_rmse_all)
            logger.info(f"Model {model}: The average {type} baseline rmse for {num_running} runs is {avg_test_rmse_all}")

    if len(test_rmse_all) > 1:
        logger.info("*"*50)
        for type in avg_rmse_for_all_models:
            logger.info(f"{type}: Average over all models for {num_running} runs "
                        f"with an rmse {np.mean(avg_rmse_for_all_models[type])}")


def get_re_from_all_langs():
    logger = create_logger("logs/all_langs.log")
    task = "wiki"
    shuffle = False
    folder = None
    model = "BLEU"

    columns = ['dataset size (sent)',
               'Source lang word TTR', 'Source lang subword TTR',
               'Target lang word TTR', 'Target lang subword TTR',
               'Source lang vocab size', 'Source lang subword vocab size',
               'Target lang vocab size', 'Target lang subword vocab size',
               'Source lang Average Sent. Length', 'Target lang average sent. length',
               'Source lang word Count', 'Source lang subword Count',
               'Target lang word Count', 'Target lang subword Count', 'geographic',
               'genetic', 'inventory', 'syntactic', 'phonological', 'featural']

    # organized data
    org_data = read_data(task, shuffle, folder, selected_feats=None)

    # length of data points
    lens = len(org_data[model]["feats"])

    # languages
    langs = org_data[model]["langs"]

    # train ids
    index = langs.index

    # test index
    for i, (source_lang, target_lang) in enumerate(langs.values):
        test_ids = langs[(langs["Source"] == source_lang) & ((langs["Target"]) == target_lang)].index
        train_ids = index.delete(test_ids)

        # splitter
        split_data = get_split_data(org_data, "specific_split", train_ids=[train_ids], test_ids=[test_ids])

        # run once
        re = get_result(split_data, "xgboost", get_ci=False, quantile=0.95)

        logger.info(f"Source Lang: {source_lang}, Target Lang: {target_lang}, rmse: {re['BLEU']['test_rmse'][0]}")

        if (i+1) % 100 == 0:
            logger.info(f"[{i+1}/{lens}] processed ")


if __name__ == '__main__':
    # paras for gpytorch
    paras = {"mean_module": {"name": "constant_mean", "paras": {}}, "covar_module": {"name": "rbf", "paras": {}}}
    monomt_features = "dataset size (sent), Source lang word TTR, Source lang subword TTR, Target lang word TTR, " \
                      "Target lang subword TTR, Source lang vocab size, Source lang subword vocab size, Target lang vocab size, " \
                      "Target lang subword vocab size, Source lang Average Sent. Length, Target lang average sent. length, " \
                      "Source lang word Count, Source lang subword Count, Target lang word Count, Target lang subword Count, " \
                      "GENETIC, SYNTACTIC, FEATURAL, PHONOLOGICAL, INVENTORY, GEOGRAPHIC".split(", ")

    # specific_evaluation(task, regressor="xgboost", get_ci=False)

    k_fold_evaluation("bli",
                      shuffle=True,
                      selected_feats=None,
                      combine_models=False,
                      regressor="xgboost",
                      k=5,
                      num_running=10)

    # get_re_from_all_langs()
