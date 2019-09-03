import numpy as np
from run_predictions import get_re_refactor, run_once, initialize_re_block, run_once_test
from train_model import calculate_rmse
from display_result import display_result, get_metric
from utils import save_pkl_file
from read_data import get_data_langs, random_split, specific_split
from task_feats import task_att

def get_result(regressor="xgboost", tasks="all", k_fold_eval=True, get_rmse=True,
               get_ci=False, quantile=0.95, standardize=False, paras=None):
    re_dict = {}
    if tasks == "all":
        tasks = ["monomt", "sf", "bli", "mi", "tsfel", "tsfmt", "tsfpos", "tsfparsing"]
    for task in tasks:
        re = get_re_refactor(task=task, regressor=regressor, k_fold_eval=k_fold_eval,
                             get_rmse=get_rmse, get_ci=get_ci, quantile=quantile, standardize=standardize,
                             paras=paras)
        re_dict[task] = re
        print("{} is done!".format(task))
    return re_dict


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
            

def get_metric_refactor(re, metric="test_rmse"):
    if metric == "test_rmse":
        for task in re:
            for eval_metric in re[task].keys():
                if eval_metric != "test_langs" or eval_metric != "test_lang_pairs":
                    reee = re[task][eval_metric]
                    print("{}_{}".format(task.capitalize(), eval_metric.capitalize()),
                          calculate_rmse(re[task][eval_metric], re[task]["{}_labels".format(eval_metric[7:])]))


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
    o = [frame.iloc[key[i]]["L1 iso code"] +'-'+ frame.iloc[key[i]]["L2 iso code"] for i in range(len(key))]
    return ', '.join(o)

def find_nbest(n=2, beam_size=5):
    paras = {"mean_module": {"name": "constant_mean", "paras": {}}, "covar_module": {"name": "rbf", "paras": {}}}
    tasks = ["mi"]
    for task in tasks:
        # Read data
        re = {}
        feats, labels, langs, lang_pairs = get_data_langs(task, shuffle=True)
        mono, _, _, _ = task_att(task)


        # Is this the correct way to get the number of items
        ids = np.arange(len(labels))
        
        # Init for @2
        beam_search_dict = {}
        best_dict = {}
        beam_search_dict[2] = {}
        best_dict[2] = []

        # Start with complete @2 for feature standardization issues
        for pair1 in ids[:10]:
            for pair2 in ids[:10]:
                if pair2 > pair1:
                    test_ids = list(ids)
                    train_ids = [pair1,pair2]
                    for i in train_ids:
                        test_ids.remove(i)
                    data = specific_split(feats, labels, lang_pairs, langs, train_ids=train_ids, test_ids=test_ids, task=task, standardize=True)
                    initialize_re_block(re, list(data.keys()), mono, langs if mono else lang_pairs)
                    for metric in data:
                        data = data[metric]
                        re = run_once_test(data, 0, metric, re, "gpytorch", get_rmse=True, get_ci=True, quantile=0.95, standardize=True, paras=paras, verbose=False)
                        #print("train_rmse:", re['Accuracy']['train_rmse'])
                        #print("test_rmse:", re['Accuracy']['test_rmse'])
                    beam_search_dict[2][(pair1,pair2)] = re['Accuracy']['test_rmse'][0]

        dict_items = list(beam_search_dict[2].items())
        keys = [k for k, v in dict_items]
        values = [v for k, v in dict_items]
        args = np.argsort(values)

        print(f"Average @2: {np.average(values):.3f} +- {np.std(values):.3f}")    
        print("Best @2")
        for i in args[:beam_size]:
            print(f"{nice_print(lang_pairs, keys[i])} : {values[i]:.3f}")
            # No need for further checks cause each tuple is unique in this case
            best_dict[2].append(keys[i])
        print("Worst @2")
        for i in args[-beam_size:]:
            print(f"{nice_print(lang_pairs, keys[i])} : {values[i]:.3f}")

        # This will only run if n>2
        for set_size in range(3,n+1):
            beam_search_dict[set_size] = {}
            best_dict[set_size] = []
            # The options to expand
            options = best_dict[set_size-1]
            for set_item in options:
                for new_item in ids[:20]:
                    if new_item not in set_item:
                        test_ids = list(ids)
                        train_ids = list(set_item) + [new_item]
                        for i in train_ids:
                            test_ids.remove(i)
                        data = specific_split(feats, labels, lang_pairs, langs, train_ids=train_ids, test_ids=test_ids, task=task, standardize=True)
                        initialize_re_block(re, list(data.keys()), mono, langs if mono else lang_pairs)
                        for metric in data:
                            data = data[metric]
                            re = run_once_test(data, 0, metric, re, "gpytorch", get_rmse=True, get_ci=True, quantile=0.95, standardize=True, paras=paras, verbose=False)
                        beam_search_dict[set_size][tuple(train_ids)] = re['Accuracy']['test_rmse'][0]

            # This could be done more efficiently on the fly but meh
            dict_items = list(beam_search_dict[set_size].items())
            keys = [k for k, v in dict_items]
            values = [v for k, v in dict_items]
            args = np.argsort(values)

            print(f"Average @{set_size}: {np.average(values):.3f} +- {np.std(values):.3f}")
            print(f"Best @{set_size}")
            cc = 0
            while len(best_dict[set_size]) < beam_size:
                ind = args[cc]
                # This check does not alow duplicates tuples (sets)
                if check_tuple(keys[ind], best_dict[set_size]):
                    print(f"{nice_print(lang_pairs, keys[ind])} : {values[ind]:.3f}")
                    best_dict[set_size].append(keys[ind])
                cc += 1
            print(f"Worst @{set_size}")
            for i in args[-beam_size:]:
                print(f"{nice_print(lang_pairs, keys[i])} : {values[i]:.3f}")
                

 
if __name__ == '__main__':
    find_nbest(n=5, beam_size=3)

