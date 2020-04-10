from src.read_data import read_data, K_Fold_Spliter, Random_Spliter, Specific_Spliter
from pathlib import Path
from src.logger import create_logger

tasks = ["wiki", "monomt", "tsfmt", "tsfparsing", "tsfpos", "tsfel", "bli", "ma", "ud"]
data_folder = Path(__file__).parent / ".."

logger = create_logger("pytest.log", vb=1)

def test_load_data():
    logger.info("*"*20)
    data = read_data(task="monomt", folder=data_folder, shuffle=True, selected_feats=None, combine_models=False)
    assert len(data["BLEU"]["feats"]) == 54
    assert len(data["BLEU"]["labels"]) == 54
    assert len(data["BLEU"]["langs"]) == 54
    assert list(data["BLEU"]["langs"].columns.values) == ["Source Language", "Target Language"]

    # test_feature_selection
    logger.info("*"*20)
    data = read_data(task="monomt", folder=data_folder, shuffle=True,
                     selected_feats=["dataset size (sent)"], combine_models=False)
    assert [data["BLEU"]["feats"].columns.values] == ["dataset size (sent)"]

    # test multi_model
    logger.info("*"*20)
    data = read_data(task="bli", folder=data_folder, shuffle=True, selected_feats=None, combine_models=False)
    assert len(data) == 3
    data = read_data(task="bli", folder=data_folder, shuffle=True, selected_feats=None, combine_models=True)
    assert len(data) == 1


def test_k_spliter():
    logger.info("*"*20)
    data = read_data(task="wiki", folder=data_folder, shuffle=True, selected_feats=None, combine_models=False)
    k_fold_spliter = K_Fold_Spliter(data)
    k_fold_data = k_fold_spliter.split()
    assert len(k_fold_data["BLEU"]["train_feats"]) == 5
    assert len(k_fold_data["BLEU"]["train_feats"][0]) + len(k_fold_data["BLEU"]["test_feats"][0]) == \
           len(k_fold_data["BLEU"]["train_feats"][1]) + len(k_fold_data["BLEU"]["test_feats"][1]) == 995


def test_random_spliter():
    logger.info("*"*20)
    data = read_data(task="ud", folder=data_folder, shuffle=True, selected_feats=None, combine_models=True)
    random_spliter = Random_Spliter(data)
    splited_data = random_spliter.split()
    assert len(splited_data["all"]["train_feats"]) == 1
    assert len(splited_data["all"]["train_feats"][0]) + len(splited_data["all"]["test_feats"][0]) == 72*25


def test_specific_spliter():
    logger.info("*"*20)
    data = read_data(task="ma", folder=data_folder, shuffle=True, selected_feats=None, combine_models=True)
    feats = data["all"]["feats"]
    lens = len(feats)
    train_idxs = list(feats[feats["data size"] > 200].index)
    test_idxs = list(set(feats.index) - set(train_idxs))
    specific_spliter = Specific_Spliter(data, train_idxs, test_idxs)
    splited_data = specific_spliter.split()
    assert len(splited_data["all"]["train_feats"][0]) + len(splited_data["all"]["test_feats"][0]) == lens
    assert len(splited_data["all"]["train_labels"][0]) + len(splited_data["all"]["test_labels"][0]) == lens
    assert len(splited_data["all"]["train_langs"][0]) + len(splited_data["all"]["test_langs"][0]) == lens






