# mono, multiple_metric, lang_vec, percent
ATT = {"tsfmt": [False, False, False, False],
       "tsfpos": [False, False, False, False],
       "tsfparsing": [False, False, False, True],
       "tsfel": [False, False, False, False],
       "mi": [False, False, False, True],
       "monomt": [False, False, False, False],
       "bli": [False, True, False, True],
       "sf": [True, True, False, False],}

def task_att(task):
    return ATT[task]

eval_metrics = {"sf": ["Keywords_Precision", "Keywords_Recall", "Keywords_F1", "NN_Precision", "NN_Recall", "NN_F1"],
                "bli": ["MUSE", "Artetxe17", "Artetxe16"],
                "monomt": ["BLEU"],
                "mi": ["Accuracy"],
                "tsfel": ["Accuracy"],
                "tsfmt": ["BLEU"],
                "tsfparsing": ["Accuracy"],
                "tsfpos": ["Accuracy"],
                "wiki": ["BLEU"]}

def task_eval_metrics(task):
    return eval_metrics[task]



