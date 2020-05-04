import sys
sys.path.append("../")

from src.read_data import read_data
import pandas as pd
from itertools import combinations
from src.run_predictions import run_once
from src.task_feats import task_eval_columns
from scipy.special import comb
from src.utils import log_args
from src.logger import create_logger
import argparse
import numpy as np
from src.train_model import calculate_rmse
from src.read_data import Specific_Spliter
from collections import defaultdict
from random import sample


def init_logging():
    logger.info("Predictions on new models running ...")
    logger.info("python3 " + " ".join(sys.argv))
    log_args(params)


def each_baseline(train_labels, test_labels):
    test_labels = test_labels.values
    test_preds = np.full_like(test_labels, train_labels.mean()[0])
    rmse = calculate_rmse(test_preds, test_labels)
    return rmse

def get_baselines(org_data, other_model_ids, sample_ids, test_ids):
    other_model_sample_ids = list(set(other_model_ids).union(set(sample_ids)))

    splitter1 = Specific_Spliter(org_data, [other_model_sample_ids], [test_ids])
    split_data1 = splitter1.split()["all"]

    splitter2 = Specific_Spliter(org_data, [sample_ids], [test_ids])
    split_data2 = splitter2.split()["all"]

    rmse1 = each_baseline(split_data1["train_labels"][0], split_data1["test_labels"][0])
    rmse2 = each_baseline(split_data2["train_labels"][0], split_data2["test_labels"][0])

    return {"other_model+new_n": rmse1, "new_n": rmse2}


def run_ex(task, n=3, regressor="xgboost", portion=0.5):
    org_data = read_data(task, True, combine_models=True)
    feats = org_data["all"]["feats"]
    ids = feats.index

    test_rmses = {}
    baseline_rmses = {}
    models = task_eval_columns(task)


    for model in models:
        logger.info("Running experiments with {} examples for a new model {}...".format(n, model))

        test_rmses[model] = []
        baseline_rmses[model] = defaultdict(list)
        model_ids = list(feats[feats[f"model_{model}"] == 1].index)
        other_model_ids = list(feats[feats[f"model_{model}"] == 0].index)

        test_lens = int(len(model_ids) * portion)

        logger.info(f"We use {portion} of the new model data as the test set. And we sample data points for training"
                    f"in the remaining {1-portion} of data. We sample the split for {params.test_id_options_num} times. "
                    f"There are {len(model_ids)} for model {model} and {len(other_model_ids)} for other models.")

        total_exs = params.test_id_options_num * params.sample_options_num
        finished_exs = 0

        for i in range(params.test_id_options_num):
            test_id_option = sample(model_ids, test_lens)

            sample_ids = list(set(model_ids) - set(test_id_option))

            total_sample_options = int(comb(len(sample_ids), n))

            logger.info("There are {} experiments running for model {}.. and we sample {} experiments".format(
                total_sample_options, model, params.sample_options_num))

            finished_exs_for_one_test_set = 0
            for j in range(params.sample_options_num):
                sample_option = sample(sample_ids, n)

                train_ids = list(set(sample_option).union(set(other_model_ids)))

                splitter = Specific_Spliter(org_data, [train_ids], [test_id_option])
                split_data = splitter.split()["all"]

                train_rmse, train_preds, test_rmse, test_preds, train_labels, test_labels, \
                    test_upper_preds, test_lower_preds, reg = \
                    run_once(split_data["train_feats"][0],
                             split_data["train_labels"][0],
                             split_data["test_feats"][0],
                             split_data["test_labels"][0],
                             split_data["train_labels_mns"][0],
                             split_data["train_labels_sstd"][0],
                             regressor,
                             get_ci=False)

                test_rmses[model].append(test_rmse)

                these_baselines = get_baselines(org_data, other_model_ids, sample_ids, test_id_option)
                for baseline in these_baselines:
                    baseline_rmses[model][baseline].append(these_baselines[baseline])

                finished_exs_for_one_test_set += 1
                finished_exs += 1
                if finished_exs % 100 == 0:
                    logger.info("Progress: {}/{}, {:.2f}%, RMSE@{}: {:.2f}".format(
                        finished_exs, total_exs, finished_exs/total_exs*100, n, np.mean(test_rmses[model])))
                    for baseline in baseline_rmses[model]:
                        logger.info(f"Baseline {baseline}: {np.mean(baseline_rmses[model][baseline])}")

                if finished_exs_for_one_test_set == params.sample_options_num:
                    break

            if finished_exs == total_exs:
                logger.info("{} done! RMSE@{}: {:.2f}".format(model, n, np.mean(test_rmses[model])))
                break

    logger.info("All experiments done!")

    for model in models:
        logger.info("Model: {}, ex: {}  RMSE@{}: {:.2f}".format(model, len(test_rmses[model]), n, np.mean(test_rmses[model])))

    logger.info("All models, RMSE@{}: {:.2f}".format(n, np.mean([np.mean(test_rmses[model]) for model in models])))
    for baseline in baseline_rmses[models[0]]:
        logger.info(f"Baseline {baseline} across all models: {np.mean([np.mean(baseline_rmses[model][baseline]) for model in models])}")


if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser(description="Models")
    parser.add_argument("--task", type=str, default="ud", help="the name of the running task")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--log", default="log", type=str, help="the log file")
    parser.add_argument("--n", type=int, default=0, help="How many should we add?")
    parser.add_argument("--portion", type=float, default=0.5, help="Portion of data points for testing.")
    parser.add_argument("--test_id_options_num", type=int, default=10, help="Number of train/test split.")
    parser.add_argument("--sample_options_num", type=int, default=10, help="Number of samples for n data points.")

    params = parser.parse_args()

    logger = create_logger(params.log, vb=params.verbose)
    init_logging()
    task = params.task

    # n = 5
    n = params.n
    run_ex(task, n, "xgboost", params.portion)




