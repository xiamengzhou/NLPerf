# mono, multiple_metric, lang_vec, percent, multiple_models
ATT = {"tsfmt": [False, False, False, False, False],
       "tsfpos": [False, False, False, False, False],
       "tsfparsing": [False, False, False, True, False],
       "tsfel": [False, False, False, False, False],
       "mi": [False, False, False, True, True],
       "monomt": [False, False, False, False, False],
       "bli": [False, True, False, True, True],
       "sf": [True, True, False, False, False],
       "wiki": [False, False, False, True, False],
       "ma": [True, True, False, True, True],
       "lemma": [True, True, False, True, True],
       "ud": [True, True, False, True, True]}

def task_att(task):
    return ATT[task]

eval_columns = {"sf": ["Keywords_Precision", "Keywords_Recall", "Keywords_F1", "NN_Precision", "NN_Recall", "NN_F1"],
                "bli": ["Sinkhorn", "Artetxe17", "Artetxe16"],
                "monomt": ["BLEU"],
                "mi": ["Accuracy"],
                "tsfel": ["Accuracy"],
                "tsfmt": ["BLEU"],
                "tsfparsing": ["Accuracy"],
                "tsfpos": ["Accuracy"],
                "wiki": ["BLEU"],
                "lemma": ["OHIOSTATE-01-2",
                          "NLPCUBE-01-2",
                          "ITU-01-2",
                          "CHARLES-SAARLAND-01-2",
                          "CHARLES-SAARLAND-02-2",
                          "CARNEGIEMELLON-02-2",
                          "CBNU-01-2",
                          "UFALPRAGUE-01-2",
                          "EDINBURGH-01-2",
                          "RUG-02-2",
                          "CMU-01-2-DataAug"],
                "ma": ["OHIOSTATE-01-2", "EDINBURGH-01-2",	"CMU-01-2-DataAug",	"CHARLES-SAARLAND-02-2",
                       "Unknown", "CARNEGIEMELLON-02-2"],
                "ud": ["HIT-SCIR", "Stanford", "TurkuNLP", "UDPipe", "ICS", "NLP-Cube", "CEA", "LATTICE", "SLT-Interactions",
                       "ParisNLP", "AntNLP", "LeisureX", "IBM", "Uppsala", "UniMelb", "KParse", "BASELINE",
                       "Fudan", "Phoenix", "CUNI", "ONLP", "BOUN"]}


def task_eval_columns(task):
    return eval_columns[task]


def get_mono(task):
    return task_att(task)[0]


def get_tasks():
    return list(ATT.keys())


def extend_metrics(task, metrics):
    return eval_columns[task].extend(metrics)
