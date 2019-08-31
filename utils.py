import pickle as pkl
import numpy as np
from copy import deepcopy
import warnings
from gp import ExactGPModel

def p2f(x):
    return float(x.strip('%'))/100


def convert_label(df):
    return df.values.reshape(len(df))

def load_pkl_file(file):
    return pkl.load(open(file, "rb"))


def save_pkl_file(ob, file, paras=None):
    ob["paras"] = paras
    pkl.dump(ob, open(file, "wb"))


def standardize_feats_df(train_feats, test_feats, return_mean_var=False):
    columns = train_feats.columns
    train_feats = deepcopy(train_feats)
    test_feats = deepcopy(test_feats)
    mns = None; sstd = None
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        c = None
        try:
            for c in columns:
                if list(set(train_feats[c].values)) == [0, 1]: # a stupid way to identify dummy variables
                    pass
                else:
                    values, mns, sstd = zscore(train_feats[c].values)
                    train_feats[c] = values
                    test_feats[c] = zscore(test_feats[c].values, mns=mns, sstd=sstd)[0]
        except Warning:
            print("Standardization goes wrong for column {} ...".format(c))
    if return_mean_var:
        assert len(columns) == 1
        return train_feats, test_feats, mns, sstd
    else:
        return train_feats, test_feats


def zscore(a, axis=0, ddof=0, mns=None, sstd=None):
    a = np.asanyarray(a)
    if mns is None:
        mns = a.mean(axis=axis)
    if sstd is None:
        sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        return ((a - np.expand_dims(mns, axis=axis)) /
                np.expand_dims(sstd, axis=axis))
    else:
        return (a - mns) / sstd, mns, sstd


def recover(mns, sstd, test_labels):
    if sstd:
        return test_labels * sstd + mns
    else:
        return test_labels


