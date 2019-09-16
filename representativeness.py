import numpy as np
from run_predictions import initialize_re_block, run_once_test
from read_data import get_data_langs, specific_split
from task_feats import task_att, task_eval_metrics
from logger import create_logger
from utils import log_args
import argparse

def init_logging():
    logger.info("Representativeness experiments running ...")
    logger.info("python3 " + " ".join(sys.argv))
    log_args(args)

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
    o = [frame.iloc[key[i], 0] +'-'+ frame.iloc[key[i], 1] for i in range(len(key))]
    return ', '.join(o)


def find_nbest(task="tsfmt", n=2, beam_size=5):
    paras = {"mean_module": {"name": "constant_mean", "paras": {}}, "covar_module": {"name": "rbf", "paras": {}}}
    tasks = [task]
    for task in tasks:
        # Read data
        re = {}
        org_data = get_data_langs(task, shuffle=True)
        mono, _, _, _ = task_att(task)
        metrics = task_eval_metrics(task)

        logger.info("Starting finding the representative languages for {}.".format(task))
        # Init for @2
        beam_search_dict = {}
        best_dict = {}
        for metric in metrics:
            metric_data = org_data[metric]
            feats = metric_data["feats"]; labels = metric_data["labels"]
            langs = metric_data["langs"]; lang_pairs = metric_data["lang_pairs"]
            lens = len(labels)
            ids = np.arange(lens)

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
            for ii, pair1 in enumerate(ids):
                for pair2 in ids[:10]:
                    if pair2 > pair1:
                        test_ids = list(ids)
                        train_ids = [pair1, pair2]
                        for i in train_ids:
                            test_ids.remove(i)
                        data = specific_split(metric_data, train_ids=train_ids, test_ids=test_ids, task=task, standardize=True)
                        initialize_re_block(re, [metric], mono, langs if mono else lang_pairs)
                        re = run_once_test(data, 0, metric, re, "gpytorch", get_rmse=True, get_ci=True, quantile=0.95, standardize=True, paras=paras, verbose=False)
                        beam_search_dict[metric][2][(pair1, pair2)] = re[metric]['test_rmse'][0]
                        finished_ex += 1
                logger.info("Progress: {}/{}, {:.2f}%".format(finished_ex, total_ex, finished_ex/total_ex*100))

            logger.info("Finished n={} experiments.".format(2))
            logger.info("Sorting {} experiments with n={}...".format(total_ex, 2))
            dict_items = list(beam_search_dict[metric][2].items())
            keys = [k for k, v in dict_items]
            values = [v for k, v in dict_items]
            args = np.argsort(values)

            logger.info(f"Average @2: {np.average(values):.3f} +- {np.std(values):.3f}")
            logger.info("Best @2")
            for i in args[:beam_size]:
                logger.info(f"{nice_print(lang_pairs, keys[i])} : {values[i]:.3f}")
                # No need for further checks cause each tuple is unique in this case
                best_dict[metric][2].append(keys[i])
            logger.info("Worst @2")
            for i in args[-beam_size:]:
                logger.info(f"{nice_print(lang_pairs, keys[i])} : {values[i]:.3f}")

            # This will only run if n > 2
            for set_size in range(3, n + 1):
                total_ex=lens-n+1; finished_ex=0
                logger.info("Searching the most representative {} languages...".format(set_size))
                logger.info("There are {} experiments running for n={}...".format(total_ex, set_size))
                # The options to expand
                options = best_dict[metric][set_size-1]
                for set_item in options:
                    for new_item in ids:
                        if new_item not in set_item:
                            test_ids = list(ids)
                            train_ids = list(set_item) + [new_item]
                            for i in train_ids:
                                test_ids.remove(i)
                            data = specific_split(metric_data, train_ids=train_ids, test_ids=test_ids, task=task, standardize=True)
                            initialize_re_block(re, list(data.keys()), mono, langs if mono else lang_pairs)
                            re = run_once_test(data, 0, metric, re, "gpytorch", get_rmse=True, get_ci=True, quantile=0.95, standardize=True, paras=paras, verbose=False)
                            beam_search_dict[metric][set_size][tuple(train_ids)] = re[metric]['test_rmse'][0]
                            finished_ex += 1
                logger.info("Progress: {}/{}, {:.2f}%".format(finished_ex, total_ex, finished_ex/total_ex*100))

                # This could be done more efficiently on the fly but meh
                dict_items = list(beam_search_dict[metric][set_size].items())
                keys = [k for k, v in dict_items]
                values = [v for k, v in dict_items]
                args = np.argsort(values)

                logger.info(f"Average @{set_size}: {np.average(values):.3f} +- {np.std(values):.3f}")
                logger.info(f"Best @{set_size}")
                cc = 0
                while len(best_dict[metric][set_size]) < beam_size:
                    ind = args[cc]
                    # This check does not allow duplicates tuples (sets)
                    if check_tuple(keys[ind], best_dict[metric][set_size]):
                        logger.info(f"{nice_print(lang_pairs, keys[ind])} : {values[ind]:.3f}")
                        best_dict[metric][set_size].append(keys[ind])
                    cc += 1
                logger.info(f"Worst @{set_size}")
                for i in args[-beam_size:]:
                    logger.info(f"{nice_print(lang_pairs, keys[i])} : {values[i]:.3f}")
                

if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser(description="Representativeness")
    parser.add_argument("--task", type=str, default="tsfmt", help="the name of the running task")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--log", type=str, help="the log file")
    parser.add_argument("--n", type=int, default=5, help="the most n representative datasets")
    parser.add_argument("--beam_size", type=int, default=3, help="the number of expanded branches when searching")
    args = parser.parse_args()

    logger = create_logger(args.log, vb=args.verbose)
    init_logging()
    task = args.task

    # n = 5
    n = args.n
    # beam_size = 3
    beam_size = args.beam_size

    find_nbest(task=task, n=n, beam_size=beam_size)
