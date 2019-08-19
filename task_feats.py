# mono, multiple_metric,
ATT = {"tsfmt": [False, False, False],
       "tsfpos": [False, False, False],
       "tsfparsing": [False, False, False],
       "tsfel": [False, False, False],
       "mi": [False, False, False],
       "monomt": [False, False, True],
       "bli": [False, True, False],
       "sf": [True, True, False],}

def task_att(task):
    return ATT[task]

eval_metrics = {"sf": ["Keywords_Precision", "Keywords_Recall", "Keywords_F1", "NN_Precision", "NN_Recall", "NN_F1"],
                "bli": ["MUSE", "Artetxe17", "Artetxe16"],
                "monomt": ["BLEU"],
                "mi": ["Accuracy"],
                "tsfel": ["Accuracy"],
                "tsfmt": ["BLEU"],
                "tsfparsing": ["Accuracy"],
                "tsfpos": ["Accuracy"]}

def task_eval_metrics(task):
    return eval_metrics[task]