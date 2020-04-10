import numpy as np
from run_predictions import run_once
from read_data import read_data, Specific_Spliter
from logger import create_logger
from utils import log_args
import argparse
import random
from collections import defaultdict


def init_logging():
    logger.info("Representativeness experiments running ...")
    logger.info("python3 " + " ".join(sys.argv))
    log_args(params)


def check_tuple(t, l):
    # Checks if tuple (t) exists in list of tuples (l)
    # But as a set (order of tuples doesn't matter)
    for item in l:
        t_set = set(t)
        i_set = set(item)
        if not i_set - t_set:
            #print(f"Found SIMILAR tuples, ignoring them: {t} exists as {item}")
            return False
    return True


def find_nbest(task="tsfmt", n=2, beam_size=5, regressor="xgboost"):
    # Read data
    re = {}
    org_data = read_data(task, shuffle=False, combine_models=True)

    logger.info("Starting finding the representative languages for {}.".format(task))
    # Init for @2
    beam_search_dict = {}
    best_dict = {}

    model_data = org_data["all"]
    feats = model_data["feats"]
    labels = model_data["labels"]
    langs = model_data["langs"]

    logger.info(f"feats shape: {feats.shape}")
    logger.info(f"labels shape: {labels.shape}")
    logger.info(f"langs shape: {langs.shape}")

    d = defaultdict(list)
    for i, (index, lang) in enumerate(zip(langs.index, langs.values)):
        d[tuple(lang)].append(index)

    ids = langs.index
    lens = len(d)
    logger.info(f"There are {lens} unique language (pairs). langs: {','.join(langs.columns)}")

    for kk in range(2, n+1):
        beam_search_dict[kk] = {}
        best_dict[kk] = []

    # Start with complete @2 for feature standardization issues
    # which n examples can give you the best predictions over the rest of all the data points

    total_ex = lens*(lens-1)/2; finished_ex=0
    logger.info("Searching the most representative 2 languages...")
    logger.info("There are {} experiments running for n={}...".format(total_ex, 2))
    for ii, pair1 in enumerate(d):
        for jj, pair2 in enumerate(d):
            if jj > ii:
                test_ids = list(ids)

                train_ids = []
                train_ids += d[pair1]
                train_ids += d[pair2]

                for i in train_ids:
                    test_ids.remove(i)

                logger.info(f"Training with {len(train_ids)} examples")
                logger.info(f"Testing with {len(test_ids)} examples")

                splitter = Specific_Spliter(org_data, [train_ids], [test_ids])
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

                beam_search_dict[2][(pair1, pair2)] = test_rmse
                finished_ex += 1
        logger.info("Progress: {}/{}, {:.2f}%".format(finished_ex, total_ex, finished_ex/total_ex*100))

    logger.info("Finished n={} experiments.".format(2))
    logger.info("Sorting {} experiments with n={}...".format(total_ex, 2))

    dict_items = list(beam_search_dict[2].items())
    keys = [k for k, v in dict_items]
    values = [v for k, v in dict_items]

    if params.type == "worst_search":
        args = np.argsort(-np.array(values))
        logger.info("Worst @2")
    elif params.type == "best_search":
        args = np.argsort(values)
        logger.info("Best @2")
    else:
        logger.info("Search type wrong. Should be best_search or worst_search.")
        sys.exit(0)

    logger.info(f"Average @2: {np.average(values):.3f} +- {np.std(values):.3f}")

    for i in args[:beam_size]:
        logger.info(f"{', '.join(['(' + ', '.join(key) + ')' for key in keys[i]])}: {values[i]:.3f}")
        # No need for further checks cause each tuple is unique in this case
        best_dict[2].append(keys[i])

    # This will only run if n > 2
    for set_size in range(3, n + 1):
        total_ex = (lens - set_size + 1) * beam_size; finished_ex = 0
        logger.info("Searching the most representative {} languages...".format(set_size))
        logger.info("There are {} experiments running for n={}...".format(total_ex, set_size))

        # The options to expand
        options = best_dict[set_size-1]
        for set_item in options:
            for kk, pair3 in enumerate(d):
                if pair3 not in set_item:
                    test_ids = list(ids)
                    train_ids = []
                    for langggg in set_item:
                        train_ids += d[langggg]
                    train_ids += d[pair3]
                    for i in train_ids:
                        test_ids.remove(i)
                    logger.info(f"Training with {len(train_ids)} examples")
                    logger.info(f"Testing with {len(test_ids)} examples")

                    splitter = Specific_Spliter(org_data, [train_ids], [test_ids])
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
                    beam_search_dict[set_size][tuple(list(set_item) + [pair3])] = test_rmse
                    finished_ex += 1
            logger.info("Progress: {}/{}, {:.2f}%".format(finished_ex, total_ex, finished_ex/total_ex*100))

        # This could be done more efficiently on the fly but meh
        dict_items = list(beam_search_dict[set_size].items())
        keys = [k for k, v in dict_items]
        values = [v for k, v in dict_items]

        if params.type == "worst_search":
            args = np.argsort(-np.array(values))
            logger.info(f"Worst @{set_size}")
        elif params.type == "best_search":
            args = np.argsort(values)
            logger.info(f"Best @{set_size}")
        else:
            logger.info("Search type wrong. Should be best_search or worst_search.")

        logger.info(f"Average @{set_size}: {np.average(values):.3f} +- {np.std(values):.3f}")
        cc = 0
        while len(best_dict[set_size]) < beam_size:
            ind = args[cc]
            # This check does not allow duplicates tuples (sets)
            if check_tuple(keys[ind], best_dict[set_size]):
                logger.info(f"{', '.join(['(' + ', '.join(key) + ')' for key in keys[ind]])}: {values[ind]:.3f}")
                best_dict[set_size].append(keys[ind])
            cc += 1


def random_search(task="tsfmt", n=2, sample=1, regressor='xgboost'):
    paras = {"mean_module": {"name": "constant_mean", "paras": {}}, "covar_module": {"name": "rbf", "paras": {}}}
    # Read data
    re = {}
    org_data = read_data(task, shuffle=False, combine_models=True)

    logger.info("Starting random search for the representative languages for {}.".format(task))
    model_data = org_data["all"]
    langs = model_data["langs"]
    ids = langs.index

    rmse = []
    finished_ex = 0
    logger.info("There are {} experiments running for n={}...".format(sample, n))
    for kk in range(sample):
        test_ids = list(ids)
        train_ids = random.sample(test_ids, n)
        for i in train_ids:
            test_ids.remove(i)

        # split data and train regressors
        splitter = Specific_Spliter(org_data, [train_ids], [test_ids])
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

        finished_ex += 1
        rmse.append(test_rmse)
        logger.info("Progress: {}/{}, {:.2f}.".format(finished_ex, sample, np.mean(rmse)))
    logger.info("Finished n={} experiments. The average rmse is {}.".format(n, np.mean(rmse)))


if __name__ == '__main__':

    import sys

    parser = argparse.ArgumentParser(description="Representativeness")
    parser.add_argument("--task", type=str, default="wiki", help="the name of the running task")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--log", default="../logs/test.log", type=str, help="the log file")
    parser.add_argument("--n", type=int, default=5, help="the most n representative datasets")
    parser.add_argument("--beam_size", type=int, default=100, help="the number of expanded branches when searching")
    parser.add_argument("--type", default="random_search", choices=["best_search", "worst_search", "random_search"], help="search type")
    params = parser.parse_args()

    logger = create_logger(params.log, vb=params.verbose)
    init_logging()
    task = params.task

    # most representative datasets search
    if params.type == "random_search":
        # random search
        for kk in range(2, 6):
            random_search(task, n=kk)
    else:
        n = params.n
        beam_size = params.beam_size
        find_nbest(task=task, n=n, beam_size=beam_size)


