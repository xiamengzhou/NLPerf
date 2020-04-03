import numpy as np
from run_predictions import initialize_re_block, run_once_test
from read_data import get_data_langs, specific_split
from task_feats import task_att, task_eval_metrics
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


def nice_print(frame, key):
    try:
        if frame.shape[1] == 2:
            o = [frame.iloc[key[i], 0] +'-'+ frame.iloc[key[i], 1] for i in range(len(key))]
        else:
            o = [frame.iloc[key[i], 0] for i in range(len(key))]
        return ",".join(o)
    except:
        return key


def find_nbest(task="tsfmt", n=2, beam_size=5):
    paras = {"mean_module": {"name": "constant_mean", "paras": {}}, "covar_module": {"name": "rbf", "paras": {}}}
    tasks = [task]
    for task in tasks:
        multi_model = task_att(task)[4]
        # Read data
        re = {}
        org_data = get_data_langs(task, shuffle=False, combine_models=True if multi_model else False)
        mono, _, _, _, _ = task_att(task)
        metrics = task_eval_metrics(task)

        logger.info("Starting finding the representative languages for {}.".format(task))
        # Init for @2
        beam_search_dict = {}
        best_dict = {}

        if multi_model:
            metrics = list(org_data.keys())
            metrics.remove("langs")
            metrics.remove("lang_pairs")

        for metric in metrics:
            metric_data = org_data[metric]
            feats = metric_data["feats"]; labels = metric_data["labels"]
            langs = metric_data["langs"]; lang_pairs = metric_data["lang_pairs"]
            logger.info("*" * 40)
            logger.info(feats.shape)
            logger.info(labels.shape)
            logger.info(langs.shape if langs is not None else "None")

            if multi_model and langs is not None:
                uni_langs = langs.values
                uni_langs = uni_langs.reshape(len(uni_langs))
                uni_langs = list(set(["_".join(l.split("_")[:-1]) for l in uni_langs]))
                lens = len(uni_langs)
                d = defaultdict(list)
                for i, lang in enumerate(langs.iloc[:, 0]):
                    d["_".join(lang.split("_")[:-1])].append(i)
            elif multi_model and lang_pairs is not None:
                uni_langs = list(set([(l1[:-2], l2[:-2]) for l1, l2 in lang_pairs.values]))
                lens = len(uni_langs)
                d = defaultdict(list)
                for i, lang_pair in enumerate(lang_pairs.values):
                    d[(lang_pair[0][:-2], lang_pair[1][:-2])].append(i)
            elif langs is not None:
                uni_langs = langs
                lens = len(langs)
                d = defaultdict(list)
                for i, lang in enumerate(langs.iloc[:, 0]):
                    d[lang].append(i)
            else:
                uni_langs = [(a, b) for a, b in lang_pairs.values]
                lens = len(lang_pairs)
                d = defaultdict(list)
                for i, lang_pair in enumerate(lang_pairs.values):
                    d[(lang_pair[0], lang_pair[1])].append(i)


            logger.info(f"There are {lens} unique languages.")
            ids = np.arange(len(labels))

            logger.info("Task: {}".format(task))
            logger.info("Metric: {}".format(metric))
            logger.info("Total number of records: {}".format(lens))
            beam_search_dict[metric] = {}
            best_dict[metric] = {}
            for kk in range(2, n+1):
                beam_search_dict[metric][kk] = {}
                best_dict[metric][kk] = []

            # Start with complete @2 for feature standardization issues
            # which n examples can give you the best predictions over the rest of all the data points

            total_ex = lens*(lens-1)/2; finished_ex=0
            logger.info("Searching the most representative 2 languages...")
            logger.info("There are {} experiments running for n={}...".format(total_ex, 2))
            for ii, pair1 in enumerate(uni_langs):
                for jj, pair2 in enumerate(uni_langs):
                    if jj > ii:
                        test_ids = list(ids)
                        train_ids = []
                        if multi_model:
                            train_ids += d[pair1]
                            train_ids += d[pair2]
                        else:
                            train_ids.append(ii)
                            train_ids.append(jj)
                        for i in train_ids:
                            test_ids.remove(i)
                        logger.info(f"Training with {len(train_ids)} examples")
                        logger.info(f"Testing with {len(test_ids)} examples")
                        data = specific_split(metric_data, train_ids=train_ids, test_ids=test_ids, task=task, standardize=False)
                        initialize_re_block(re, [metric], mono, langs if mono else lang_pairs)
                        re = run_once_test(data, 0, metric, re, "xgboost", get_rmse=True, get_ci=False, quantile=0.95, standardize=False, paras=paras, verbose=False)
                        beam_search_dict[metric][2][(pair1, pair2)] = re[metric]['test_rmse'][0]
                        finished_ex += 1
                logger.info("Progress: {}/{}, {:.2f}%".format(finished_ex, total_ex, finished_ex/total_ex*100))

            logger.info("Finished n={} experiments.".format(2))
            logger.info("Sorting {} experiments with n={}...".format(total_ex, 2))
            dict_items = list(beam_search_dict[metric][2].items())
            keys = [k for k, v in dict_items]
            values = [v for k, v in dict_items]
            if params.worst:
                args = np.argsort(-np.array(values))
            else:
                args = np.argsort(values)

            langs_langs_pairs = uni_langs if mono else lang_pairs
            logger.info(f"Average @2: {np.average(values):.3f} +- {np.std(values):.3f}")
            logger.info("Best @2")
            for i in args[:beam_size]:
                logger.info(f"{nice_print(langs_langs_pairs, keys[i])} : {values[i]:.3f}")
                # No need for further checks cause each tuple is unique in this case
                best_dict[metric][2].append(keys[i])
            logger.info("Worst @2")
            for i in args[-beam_size:]:
                logger.info(f"{nice_print(langs_langs_pairs, keys[i])} : {values[i]:.3f}")

            # This will only run if n > 2
            for set_size in range(3, n + 1):
                total_ex=(lens - set_size + 1) * beam_size; finished_ex=0
                logger.info("Searching the most representative {} languages...".format(set_size))
                logger.info("There are {} experiments running for n={}...".format(total_ex, set_size))
                # The options to expand
                options = best_dict[metric][set_size-1]
                for set_item in options:
                    for kk, pair3 in enumerate(uni_langs):
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
                            data = specific_split(metric_data, train_ids=train_ids, test_ids=test_ids, task=task, standardize=False)
                            initialize_re_block(re, list(data.keys()), mono, langs if mono else lang_pairs)
                            re = run_once_test(data, 0, metric, re, "xgboost", get_rmse=True, get_ci=False, quantile=0.95, standardize=False, paras=paras, verbose=False)
                            beam_search_dict[metric][set_size][tuple(list(set_item) + [pair3])] = re[metric]['test_rmse'][0]
                            finished_ex += 1
                    logger.info("Progress: {}/{}, {:.2f}%".format(finished_ex, total_ex, finished_ex/total_ex*100))

                # This could be done more efficiently on the fly but meh
                dict_items = list(beam_search_dict[metric][set_size].items())
                keys = [k for k, v in dict_items]
                values = [v for k, v in dict_items]
                if params.worst:
                    args = np.argsort(-np.array(values))
                else:
                    args = np.argsort(values)

                logger.info(f"Average @{set_size}: {np.average(values):.3f} +- {np.std(values):.3f}")
                logger.info(f"Best @{set_size}")
                cc = 0
                while len(best_dict[metric][set_size]) < beam_size:
                    ind = args[cc]
                    # This check does not allow duplicates tuples (sets)
                    if check_tuple(keys[ind], best_dict[metric][set_size]):
                        logger.info(f"{nice_print(langs_langs_pairs, keys[ind])} : {values[ind]:.3f}")
                        best_dict[metric][set_size].append(keys[ind])
                    cc += 1
                logger.info(f"Worst @{set_size}")
                for i in args[-beam_size:]:
                    logger.info(f"{nice_print(langs_langs_pairs, keys[i])} : {values[i]:.3f}")

def random_search(task="tsfmt", n=2, sample=100):
    paras = {"mean_module": {"name": "constant_mean", "paras": {}}, "covar_module": {"name": "rbf", "paras": {}}}
    tasks = [task]
    for task in tasks:
        # Read data
        re = {}
        org_data = get_data_langs(task, shuffle=False)
        mono, _, _, _ = task_att(task)
        metrics = task_eval_metrics(task)

        logger.info("Starting random search for the representative languages for {}.".format(task))
        for metric in metrics:
            metric_data = org_data[metric]
            feats = metric_data["feats"]
            labels = metric_data["labels"]
            langs = metric_data["langs"]
            lang_pairs = metric_data["lang_pairs"]
            lens = len(labels)
            ids = np.arange(lens)

            logger.info("Task: {}".format(task))
            logger.info("Metric: {}".format(metric))
            logger.info("Total number of records: {}".format(lens))

            rmse = []
            finished_ex = 0
            logger.info("There are {} experiments running for n={}...".format(sample, n))
            for kk in range(sample):
                test_ids = list(ids)
                train_ids = random.sample(test_ids, n)
                for i in train_ids:
                    test_ids.remove(i)
                data = specific_split(metric_data, train_ids=train_ids, test_ids=test_ids, task=task,
                                      standardize=False)
                initialize_re_block(re, [metric], mono, langs if mono else lang_pairs)
                re = run_once_test(data, 0, metric, re, "xgboost", get_rmse=True, get_ci=False, quantile=0.95,
                                   standardize=False, paras=paras, verbose=False)
                finished_ex += 1
                rmse.append(re[metric]['test_rmse'][0])
                logger.info("Progress: {}/{}, {:.2f} for {}.".format(finished_ex, sample, np.mean(rmse), metric))
            logger.info("Finished n={} experiments. The average rmse for metric {} is {}.".format(n, metric, np.mean(rmse)))

if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser(description="Representativeness")
    parser.add_argument("--task", type=str, default="monomt", help="the name of the running task")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--log", type=str, help="the log file")
    parser.add_argument("--n", type=int, default=5, help="the most n representative datasets")
    parser.add_argument("--beam_size", type=int, default=100, help="the number of expanded branches when searching")
    parser.add_argument("--worst", action="store_true", help="best or worst")
    params = parser.parse_args()

    logger = create_logger(params.log, vb=params.verbose)
    init_logging()
    task = params.task

    # n = 5
    n = params.n
    # beam_size = 3
    beam_size = params.beam_size

    find_nbest(task=task, n=n, beam_size=beam_size)
    # for kk in range(2, 6):
    #     random_search(task, n=kk)
