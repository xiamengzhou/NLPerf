from train_model import calculate_rmse, calculate_mean_bounds
import pyperclip

def get_metric(re):
    metric_re = {}
    for task in re:
        if task != "paras":
            metric_re[task] = {}
            for metric in re[task]:
                # should retrieve metrics from list of metrics
                if metric != "ori_test_lang_pairs" and metric != "ori_test_langs":
                    metric_re[task][metric] = {}
                    ree = re[task][metric]
                    test_rmse = calculate_rmse(ree["result_{}".format(metric)], ree["{}_labels".format(metric)])
                    metric_re[task][metric]["test_rmse"] = test_rmse
                    metric_re[task][metric]["mean_bounds"] = calculate_mean_bounds(ree["{}_lower_preds".format(metric)],
                                                       ree["{}_upper_preds".format(metric)])
    return metric_re


def update_metric_re(metric_dicts):
    for di in metric_dicts[1:]:
        for task in di:
            metric_dicts[0][task].update(di[task])
    return metric_dicts[0]


def display_result(re, tasks=None, metrics=None):
    def display_one(task, metric, l):
        print(l)
        print()
    if not tasks:
        tasks = list(re.keys())
    for task in sorted(tasks):
        print(task)
        for metric in sorted(re[task]):
            for i, e in enumerate(re[task][metric]):
                if i == 0:
                    print(metric, "%.2f" % re[task][metric][e], end=" ")
                else:
                    print("%.2f" % re[task][metric][e])








