import pandas as pd
import lang2vec.lang2vec as l2v
import os
import sys
from src.task_feats import task_eval_metrics
import glob
import conllu
from legacy.unimorph import get_tag_to_type, get_type_to_tag, get_all_tags
from collections import Counter, defaultdict

langvec_lens = {"syntax_average": 103,
                "phonology_average": 28,
                "inventory_average": 158,
                "geo": 299,
                "fam": 3718}

if len(sys.argv) == 1:
    sys.argv.append("")

def load_treebank(dir, lang):
    try:
        path = os.path.join(dir, lang, "*train.conllu")
        file = list(glob.glob(path))[0]
        return conllu.parse(open(file, "r").read())
    except:
        return None

def extract_lemma_feats(data):
    word_lemmas = defaultdict(set)
    for sent in data:
        for word in sent:
            word_lemmas[word["form"]].add(word["lemma"])
    num_of_words = len(word_lemmas)
    average_type_lemma_for_word = sum([len(word_lemmas[word]) for word in word_lemmas]) / len(word_lemmas)
    feat = {}
    feat["num of words"] = num_of_words
    feat["average type lemma for word"] = average_type_lemma_for_word
    return feat

def get_distance(src, tgt):
    d1 = l2v.genetic_distance(src, tgt)
    d2 = l2v.syntactic_distance(src, tgt)
    d3 = l2v.featural_distance(src, tgt)
    d4 = l2v.phonological_distance(src, tgt)
    d5 = l2v.inventory_distance(src, tgt)
    d6 = l2v.geographic_distance(src, tgt)
    return d1, d2, d3, d4, d5, d6

def extract_tag_feats(data, all_tags):
    fusion_tags = []
    tags = []
    word_num = sum([len(sent) for sent in data])
    words = []
    word_tags = defaultdict(set)
    data_size = len(data)
    for sent in data:
        for word in sent:
            feats = word["feats"]
            words.append(word["form"])
            if feats is not None:
                fusion_tag = list(feats.keys())[0]
                fusion_tags.append(fusion_tag)
                tags.extend(fusion_tag.split(";"))
                word_tags[word["form"]].add(list(feats.keys())[0])
            else:
                word_tags[word["form"]].add(None)
    tag_per_word = len(tags) / word_num
    num_type_fusion_tag = len(set(fusion_tags))
    num_type_tag = len(set(tags))
    counter = Counter(tags)
    tag_freq = {}
    for tag in all_tags:
        tag_freq[tag] = (counter.get(tag) if counter.get(tag) is not None else 0) / word_num
    average_tag_type_length = sum([len(get_type_to_tag(get_tag_to_type(tag)))*
                                   (counter.get(tag) if counter.get(tag) is not None else 0) for tag in all_tags]) / word_num
    average_type_tag_for_word = sum([len(word_tags[word]) for word in word_tags]) / len(word_tags)
    feat = {"tag per word": tag_per_word,
            "num type fusion tag": num_type_fusion_tag,
            "num type tag": num_type_tag,
            "average tag type length": average_tag_type_length,
            "average type tag for word": average_type_tag_for_word,
            "word num": word_num,
            "word type": len(set(words)),
            "data size": data_size,
            "avg sent length": word_num / data_size}
    for tag in tag_freq:
        feat[f"{tag} freq"] = tag_freq[tag]
    return feat

def augment_ud_feats():
    dir = "/projects/tir2/users/mengzhox/data/ud/ud-treebanks-v2.4"
    d = {}
    lines = open(os.path.join(dir, "langs"), "r").readlines()
    for line in lines:
        UD, ud = line.split("/")
        ud = ud.split("-")[0]
        d[ud] = UD
    all_tags = get_all_tags()
    df = pd.read_csv(f"data/data_ud.tsv", sep="\t")
    df_labels = df.loc[:, task_eval_metrics("ud")]
    df = df.drop(axis=1, columns=task_eval_metrics("ud"))
    removed_index = []
    for i, lang in zip(df.index, df["Language"]):
        if lang in d:
            data = load_treebank(dir, d[lang])
            if data is None:
                continue
            feat = extract_tag_feats(data, all_tags)
            for key in feat:
                df.loc[i, key] = feat[key]
        else:
            removed_index.append(i)
        print("Finished {}".format(lang))
    df = df.drop(axis=0, index=removed_index)
    df_labels = df_labels.drop(axis=0, index=removed_index)
    df = pd.concat([df, df_labels], axis=1)
    df.to_csv(os.path.join("data", "data_{}_new.csv".format("ud")))



def augment_feats(task):
    dir = os.path.join("/Users/mengzhouxia/Git", "2019", "task2")
    df = pd.read_csv(os.path.join(sys.argv[1], "data", "data_{}.csv".format(task)))
    df_labels = df.loc[:, task_eval_metrics(task)]
    df = df.drop(axis=1, columns=task_eval_metrics(task))

    all_tags = get_all_tags()

    for i, lang in zip(df.index, df["language"]):
        data = load_treebank(dir, lang)
        if task == "ma":
            feat = extract_tag_feats(data, all_tags)
        elif task == "lemma":
            feat = extract_lemma_feats(data)

        else:
            sys.exit(0)
        for key in feat:
            df.loc[i, key] = feat[key]
        print("Finished {}".format(lang))
    df = pd.concat([df, df_labels], axis=1)
    df.to_csv(os.path.join(sys.argv[1], "data", "data_{}_new.csv".format(task)))


def augment_langvec(task, column):
    feats = ["syntax_average", "phonology_average", "inventory_average", "geo", "fam"]
    dims = [103, 28, 158, 299, 158]

    df = pd.read_csv(os.path.join(sys.argv[1], "data", "data_{}.csv".format(task)))

    for i, feat in enumerate(feats):
        for j in range(langvec_lens[feat]):
            df[feat + "_{}".format(j)] = 0

    for index, lang in zip(df.index, df.loc[:, column]):
        features = l2v.get_features(lang, "+".join(feats))[lang]
        print("Loading langvec for {} ..".format(lang))
        for j, feat in enumerate(feats):
            start = sum(dims[:j])
            end = sum(dims[:j+1])
            df.loc[index, [feat + "_" + str(i) for i in range(dims[j])]] = features[start: end]
    df.to_csv(os.path.join(sys.argv[1], "data/data_lemma_new.csv".format(task)), index=0)

def swap_columns(task):
    df = pd.read_csv(os.path.join(sys.argv[1], "data", "data_{}_new.csv".format(task)), index_col=0)
    metrics = task_eval_metrics(task)
    labels = df.loc[: , metrics]
    df.drop(columns=metrics, inplace=True)
    df = pd.concat([df, labels], axis=1)
    # type conversion
    for i, feat in enumerate(langvec_lens.keys()):
        for j in range(langvec_lens[feat]):
            ff = feat + "_{}".format(j)
            se = df[feat + "_{}".format(j)]
            # else if contains nan?
            if se.where(se == "--").any():
                df.drop(columns=[ff], inplace=True)
            else:
                unique_values = df[feat + "_{}".format(j)].unique()
                # if unique value?
                if len(unique_values) == 0:
                    df.drop(columns=[ff], inplace=True)
    df.to_csv(os.path.join(sys.argv[1], "data", "data_{}_new2.csv".format(task)), index=0)

def augment_lang(task):
    cols = ["GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]
    df = pd.read_csv(os.path.join(sys.argv[1], "data", "data_{}.csv".format(task)), index_col=0)
    index = df.index
    for i, (src, tgt) in enumerate(df.iloc[:, :2].values):
        ds = get_distance(src, tgt)
        for j, d in enumerate(ds):
            df.loc[index[i], cols[j]] = d
        print(i)
    df.to_csv(os.path.join(sys.argv[1], "data", "data_{}_new2.csv".format(task)), index=0)

def swap_langs(task):
    df = pd.read_csv(os.path.join(sys.argv[1], "data", "data_{}_new2.csv".format(task)))
    bleu = df["BLEU"]
    df = df.drop(labels=["BLEU"], axis=1)
    df["BLEU"] = bleu
    df.to_csv(os.path.join(sys.argv[1], "data", "data_{}_new3.csv".format(task)))

def augment_mt_feats():
    df = pd.read_csv("data/data_tsfmt.csv")
    feats = df.columns[-6:]
    target_langs = df["Target lang"].unique()
    import pickle
    if os.path.exists("data/monomt_feats.pkl"):
        haha = pickle.load(open("data/monomt_feats.pkl", "rb"))
    else:
        haha = {}
        for tl in target_langs:
            lang_feats = get_distance(tl, "eng")
            haha[tl] = lang_feats
            print("Collected", tl)
        pickle.dump(haha, open("data/monomt_feats.pkl", "wb"))

    for i, index in enumerate(df.index):
        for j, feat_name in enumerate(feats):
            df.loc[index, feat_name + "_2"] = haha[df.loc[index, "Target lang"]][j]
    df.to_csv("data/data_tsfmt.csv", index=False)

def augment_bli_features():
    langs = pd.read_csv("data/data_bli.csv")["Target Language Code"].unique()
    df = pd.DataFrame(columns=["syntax_" + str(i) for i in range(103)])
    for lang in langs:
        features = l2v.get_features(lang, l2v.fs_concatenation(
            [l2v.fs_union(["syntax_wals", "syntax_sswl", "syntax_ethnologue"])]))
        df.loc[lang] = features[lang]
        print(lang + " done!")
    df.to_csv("data/bli_tgt_feats.csv")

# augment_langvec("lemma", "language code")
# augment_feats("ma")
# augment_mt_feats()
augment_bli_features()