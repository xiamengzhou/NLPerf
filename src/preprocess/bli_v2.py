import pandas as pd
from collections import defaultdict
import os
import lang2vec.lang2vec as l2v
import numpy as np

lang_codes = {"az": "aze", "be": "bel", "cs": "ces", "en": "eng", "es": "spa", "pt": "por", "ru": "rus",
              "sk": "slk", "tr": "tur", "fr": "fra", "hi": "hin", "ko": "kor", "sv": "swe", "uk": "ukr",
              "gl": "glg"}

def convert_data_frame(dfs):
    d = defaultdict(list)
    for key in dfs:
        df = dfs[key]
        # manipulation over row data
        df_data = pd.read_csv(df, index_col=0)
        df_data1 = df_data.iloc[:10, :10]
        df_data2 = df_data.iloc[11:19, :7]
        df_data2.columns = df_data2.iloc[0, :]
        df_data2 = df_data2.iloc[1:]
        for dfss in [df_data1, df_data2]:
            for i, src in enumerate(dfss.index):
                for j, tgt in enumerate(dfss.columns):
                    if i != j and not dfss.loc[src, tgt] == "--":
                        d[f"{src}-{tgt}"].append(float(dfss.loc[src, tgt]))
    data = []
    for key in d:
        src, tgt = key.split("-")
        d[key].insert(0, src); d[key].insert(1, tgt)
        data.append(d[key])
    return pd.DataFrame(data, columns=["Source Language Code", "Target Language Code"] + list(dfs.keys()))


def add_lang_distance(df):
    lang_feats = "GENETIC,SYNTACTIC,FEATURAL,PHONOLOGICAL,INVENTORY,GEOGRAPHIC".lower().split(",")

    dists = []
    for src, tgt in df[["Source Language Code", "Target Language Code"]].values:
        src = lang_codes[src]; tgt = lang_codes[tgt]
        distance = l2v.distance(lang_feats, src, tgt)
        dists.append(distance)
        print(f"Collected distance for {src} and {tgt}.")
    dists = np.array(dists).transpose()

    for lang_feat, dist in zip(lang_feats, dists):
        df[lang_feat] = dist

    return df


def add_syntax_feats(df):
    langs = set(df["Target Language Code"]).union(df["Source Language Code"])
    source_columns = ["syntax_src_" + str(i) for i in range(103)]
    target_columns = ["syntax_tgt_" + str(i) for i in range(103)]

    for column in source_columns + target_columns:
        df[column] = 0

    # collect syntax features for all langs
    d = {}
    for lang in langs:
        lang = lang_codes[lang]
        features = l2v.get_features(lang, l2v.fs_concatenation(
            [l2v.fs_union(["syntax_wals", "syntax_sswl", "syntax_ethnologue"])]))
        d[lang] = features[lang]

    # add all syntax features to dataframe
    for index in df.index:
        src = df.loc[index, "Source Language Code"]
        tgt = df.loc[index, "Target Language Code"]
        df.loc[index, source_columns] = d[lang_codes[src]]
        df.loc[index, target_columns] = d[lang_codes[tgt]]
        print(f"Collected syntactic features for {src} and {tgt}.")

    return df


if __name__ == '__main__':
    data_dir = "../../data/raw_data"
    dfs = {'Muse': os.path.join(data_dir, 'BLI Results - BLI Muse.csv'),
           'Vecmap': os.path.join(data_dir, 'BLI Results - BLI Vecmap.csv')}
    df = convert_data_frame(dfs)

    df = add_lang_distance(df)
    df = add_syntax_feats(df)

    df.to_csv(os.path.join(data_dir, "data_bli2.csv"))
