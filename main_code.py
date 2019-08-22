from xgboost import plot_importance
import numpy as np
from run_predictions import get_re_refactor
from train_model import calculate_rmse

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

def plot(ree, type="weight", lang="ara", task="mt"):
    task_re = ree[task]
    print(task_re["rmse"][lang])
    ax = plot_importance(task_re["reg"][lang], importance_type=type)
    fig = ax.figure
    fig.set_size_inches(5, 30)

def get_metric(re, metric="test_rmse"):
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




# parameters -> settings
# how can I find the best options among different parameters
# add which experiment results there, can we get the best prediction power for the rest of the experiments
# what are the 5 most accurate predictions given a prediction model
# how can we do evaluation to verify that these 5 predictions are indeed the most useful ones?
# for a new-coming language, we want to relate the language back to previous ones
# get_result(regressor="gb", tasks="monomt", get_ci=True)
