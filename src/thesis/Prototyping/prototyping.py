"""
Atom Hydrogen / VSCode Python compatible notebook variant used for testing and prototyping code
"""
#%%
import os
import os.path
import pickle
import sys
from functools import partial
from multiprocessing import Manager, Pool
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

%matplotlib inline

#%% util
def print_table(cols, data, max_width=None, prefix=None, suffix=None):
    widths = [max_width if max_width else 1e309 for _ in cols]
    for idx, width in enumerate(widths):
        def my_len(x): return len(f"{x:.3f}" if isinstance(
            x, (float, np.inexact)) else str(x))
        content_length = max(len(cols[idx]), max(
            [my_len(tpl[idx]) for tpl in data]))
        widths[idx] = min(width, content_length) + 2

    table = ""
    for idx, col in enumerate(cols):
        arr = "<" if idx == 0 else ">"
        table += f"{col:{arr}{widths[idx]}}|"
    table += "\n"
    for idx, _ in enumerate(cols):
        table += f"{'':-<{widths[idx]}}+"
    table += "\n"

    for tpl in data:
        for idx, entry in enumerate(tpl):
            width = widths[idx]
            if isinstance(entry, (float, int, np.number)):
                if isinstance(entry, (float, np.inexact)):
                    s = f"{entry:.3f}"
                elif isinstance(entry, (int, np.integer)):
                    s = f"{entry:d}"
                if len(s) > (width - 1):
                    s = f"{entry:.3e}"
            else:
                s = str(entry)
                if len(s) > (width - 1):
                    s = s[:width-4] + "..."

            arr = "<" if idx == 0 else ">"
            table += f"{s:{arr}{width}}|"

        table += "\n"

    print(table)


def mp_drop_na(df_name, data_dict):
    """
    Function to drop all columns from df_name in data_dict that only contain missing values or otherwise coded as non-values.
    """
    df = data_dict[df_name]
    no_cols = len(df.columns)
    # missing values are coded as either NA or negative numbers
    cols_to_keep = df.columns[(df.notna() & (df >= 0)).sum() != 0]
    data_dict[df_name] = df[cols_to_keep]
    no_dropped_cols = no_cols - len(cols_to_keep)

    return df_name, no_dropped_cols


def clean(data_dict, data_dict_path="./b_PreProcessing/Data/data_dict_cleaned"):
    """
    Cleaning the data by dropping or fixing values/columns.
    """
    if _pool and _manager:
        func = partial(mp_drop_na, data_dict=data_dict)
        result = list(_pool.map(func, list(data_dict.keys())))
    else:
        result = []
        for df_name in data_dict.keys():
            result.append(mp_drop_na(df_name, data_dict))

    for df_name, no_dropped in result:
        print(f"{df_name:<20}{no_dropped:>4} dropped")

    # save_data_dict(data_dict_path, data_dict)


def save_pickle(obj, filename):
    with open(filename, mode="wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(filename):
    with open(filename, mode="rb") as fh:
        obj = pickle.load(fh)
    return obj


def load_files():
    valid_ids_path = "../b_PreProcessing/Data/valid_ids"
    data_dict_path = "../b_PreProcessing/Data/data_dict"
    cleaned_data_dict_path = "../b_PreProcessing/Data/data_dict_cleaned"
    cleaned_data_dict_path = ""

    try:
        valid_ids = load_pickle(valid_ids_path)
        if os.path.isfile(cleaned_data_dict_path):
            data_dict = load_pickle(cleaned_data_dict_path)
        else:
            data_dict = load_pickle(data_dict_path)
            clean(data_dict)
    except IOError as e:
        print("Could not load file", e, sep="\n", file=sys.stderr)

    print(valid_ids.shape, len(data_dict))
    return valid_ids, data_dict


def change_work_dir():
    """
    Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
    """
    import os
    try:
        os.chdir(os.path.join(os.getcwd(), 'src/Prototyping'))
        print(os.getcwd())
    except Exception as e:
        print(e)


def count_id_occurence():
    cols, lst = ["Table", "ID max"], []
    for df_k in data_dict.keys():
        df = data_dict[df_k]
        m = df["ID_t"].value_counts().max()
        lst.append((df_k, m))

    lst.sort(key=lambda x: x[1], reverse=True)
    print_table(cols, lst)


def build_variable_tag_set():
    d = {}
    for df_k in data_dict.keys():
        cols = data_dict[df_k].columns
        for col in cols:
            tag = col[:2]
            if tag in d and not col in d[tag]:
                d[tag].add(col)
            else:
                d[tag] = {col}

    return d


def plot_study_dates():
    df = keys.merge(right=vocTrain, how="left", on=["ID_t", "spell"])
    times = df[["ID_t", "spell", "wave", "ts15218", "ts1511m_g1", "ts1511y_g1", "ts1512m_g1", "ts1512y_g1"]]
    times = times[~(times.T.isna() | (times.T < 0)).any()]
    times["start"] = pd.to_datetime(times["ts1511y_g1"].astype(int).astype(str) + "-" + times["ts1511m_g1"].astype(int).astype(str) + "-1")
    times["end"] = pd.to_datetime(times["ts1512y_g1"].astype(int).astype(str) + "-" + times["ts1512m_g1"].astype(int).astype(str) + "-1")
    times = times.drop(["ts1511m_g1", "ts1511y_g1", "ts1512m_g1", "ts1512y_g1"], axis='columns')
    # times.groupby("ts15218").agg(["max", "min", "mean", lambda x: x.value_counts().index[0]])
    succ = times[times["ts15218"] == 1]
    succ.name = "Successful studies"
    drop = times[times["ts15218"] == 2]
    drop.name = "Dropped out"
    
    start = np.datetime64("2000-01-02")
    succ[succ["start"] < start].merge(right=CATI[["ID_t", "t70000m", "t70000y"]], on=["ID_t"], how="left"),\
    drop[drop["start"] < start].merge(right=CATI[["ID_t", "t70000m", "t70000y"]], on=["ID_t"], how="left")

    start = np.datetime64("2000-01-01")
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(1,1,1)
    x_succ = succ["start"][succ["start"] > start]
    # y_succ = (succ["end"][succ["start"] > start] - x)/ np.timedelta64(1, 'Y')
    # ax.scatter(x, y, color="green", label="succ", s=25)

    x_drop = drop["start"][drop["start"] > start]
    # y_drop = (drop["end"][drop["start"] > start] - x)/ np.timedelta64(1, 'Y')
    # ax.scatter(x, y, color="red", label="drop", s=25)

    bins = 15
    arr = ax.hist([x_succ, x_drop], bins=bins, log=False, density=False, rwidth=.8, label=["succ", "drop"])

    for i in range(bins):
        ax.text(arr[1][i], arr[0][0][i], str(int(arr[0][0][i])), color="blue", fontsize=12)
        ax.text(arr[1][i]+135, arr[0][1][i], str(int(arr[0][1][i])), fontsize=12)

    # ax = sns.distplot(x_succ)#, kde=False)#, norm_hist=True)

    ax.set_xlabel("Start of Studies")
    ax.set_ylabel("Relative Amount")
    # # ax.set_ylim((0, 10))
    ax.legend()


def voc_id_spells():
    vocTrain = data_dict["spVocTrain"]
    CATI = data_dict["pTargetCATI"]

    variables = [
        "ID_t",  # target id
        "wave",  # wave of episode
        "spell",  # spell id
        "ts15221_v1",  # angestrebter Ausbildungsabschluss
        "ts15219_v1",  # Ausbildungsabschluss: B.Sc, M.Sc, Ausbildung, etc.
        "ts15265",  # Note des Abschluss
        "tg24103",  # Episodenmodus
        "ts15201",  # Ausbildungstyp
        "ts15218",  # Erfolgreicher Abschluss
        "tg24159",  # Fachwechsel gegenüber Vorepisode; 1=same, 2=diff
        "tg24121",  # Hochschulwechsel gegenüber Vorepisode; 1=same, 2=diff
        "ts1512c",  # Andauern der Episode
        "ts1511m_g1",  # Prüfmodul: Startdatum (Monat, ediert)
        "ts1511y_g1",  # Prüfmodul: Startdatum (Jahr, ediert)
        "ts1512m_g1",  # Prüfmodul: Enddatum (Monat, ediert)
        "ts1512y_g1",  # Prüfmodul: Enddatum (Jahr, ediert)
    ]

    subset = vocTrain[
        (vocTrain["subspell"] == 0)
        & (vocTrain["disagint"] != 1)
    ][variables]

    relevant_education = [
        10,  # Studium an einer Universität, auch pädagogische Hochschule, Kunst- und Musikhochschule
        # Studium an einer Fachhochschule, auch Hochschule für angewandte Wissenschaften oder University of Applied Sciences genannt (nicht Verwaltungsfachhochschule
        9,
        -28  # Wert aus Rekrutierung pTargetCATI ## might be relevant
    ]
    relevant_courses_of_study = [
        13,  # BA ohne Lehramt
        #17, # Erstes Staatsexamen Lehramt
        8,  # BA
        12,  # BA Lehramt
        #-54, # Designbedingt fehlend
    ]
    curr = subset[
        # filter for uni/fh only
        subset["ts15201"].isin(relevant_education) &
        subset["ts15221_v1"].isin(
            relevant_courses_of_study)  # filter bachelor only
    ]
    curr["ts15218"].value_counts()

    # prepare subset of CATI for joining
    other = CATI[
        # the question was only asked in wave 1, and this way ID_t becomes the unique identifier
        CATI["wave"] == 1
        & CATI["tg02001"].isin([1, 3])  # filter for 3=BA and 1=BA Lehramt only
    ][["tg02001", "ID_t"]]  # ID_t to join on, tg02001 holds the pursued degree

    # join CATI subset into dataframe for more complete sample
    # integrate tg02001 into df to fill in ts15201 -28
    curr = curr.join(other.set_index("ID_t"), on="ID_t")
    # filter out those rows that CATI added nothing too, meaning those without information on pursued degree even after joining
    curr = curr[curr["ts15201"] != -28 |
                (curr["ts15201"] == -28 & ~curr["tg02001"].isnull())]

    # reorder columns in tmp dataframe for a nicer overview, moving joined CATI degree next to vocTrain degree
    cols = list(curr.columns)
    purs_deg_idx = cols.index("ts15221_v1") + 1
    cols = cols[:purs_deg_idx] + cols[-1:] + cols[purs_deg_idx:-1]
    curr = curr[cols]

    # filter out all students who have one spell with an unsuccesfull graduation
    dropped_students = curr[(curr["ts15218"] == 2)]["ID_t"].unique()
    dropped = curr[curr["ID_t"].isin(dropped_students)]

    # check how many samples we have per condition
    ## dropped out -> no further spell of interest
    dropped1 = dropped[
        (dropped["ts15218"] == 2)
        & (
            dropped["ID_t"].isin(
                dropped["ID_t"].value_counts(
                )[dropped["ID_t"].value_counts() == 1].index
            )
        )
    ]

    ## dropped out -> spell of interest
    ### -> changed subject, irrelevant of change of institution
    # find all those who failed and have any preluding or subsequent spell
    tmp = dropped[
        (dropped["ts15218"] == 2) &
        (dropped["ID_t"].isin(dropped["ID_t"].value_counts()
                                [dropped["ID_t"].value_counts() > 1].index))
    ]
    # shift up all rows by one, then select those indices we filtered above
    # this gives us the possibility to compare two subsequent rows using the same index
    dropped2 = dropped.shift(-1).loc[tmp.index]
    # We only want to compare rows for the same student, and then filter those who actually changed subject
    # leaving out those who only switched institutions or had other reasons for ending their spell
    # and also leaving out possible mismatches, where all recorded spells of interest were unsuccesfull,
    # and thus the following row is already for the next student
    dropped2 = dropped2[
        (dropped2["ID_t"] == tmp["ID_t"]) &
        (dropped2["tg24159"] == 2)
    ]

    keys = dropped1[["ID_t", "spell"]].append(dropped2[["ID_t", "spell"]]).append(curr[curr["ts15218"] == 1][["ID_t", "spell"]])
    return keys


def forest_feature_importance(dd, valid_ids, df):
    succ = valid_ids.merge(dd["spVocTrain"][["ID_t", "spell", "ts15218"]], on=["ID_t", "spell"])
    succ = succ.dropna().drop_duplicates("ID_t").drop(columns="spell")
    if type(df) is str:
        df = dd[df]
    eval = df.merge(succ, on="ID_t")
    Y = eval["ts15218"]
    eval.drop(columns=["ts15218", "ID_t"], inplace=True)
    eval[eval.isna()] = -1
    xTrain, xTest, yTrain, yTest = train_test_split(eval, Y, test_size=0.2)

    clf_rfc = RFC(n_estimators=5000)
    clf_rfc = clf_rfc.fit(xTrain, yTrain)
    print(f"Forest scored: {clf_rfc.score(xTest, yTest):.4f}")

    importances = clf_rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rfc.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")

    for f in range(xTrain.shape[1]):
        print(
            f"{f + 1}. feature {xTrain.columns[indices[f]]} ({importances[indices[f]]:.4f})")

    plt.figure(figsize=(28,18))
    plt.gca().set_facecolor("xkcd:salmon")
    plt.title("Feature importances")
    plt.bar(range(xTrain.shape[1]), importances[indices],
        color="g", yerr=std[indices], align="center")
    plt.xticks(range(xTrain.shape[1]), indices)
    plt.xlim([-1, xTrain.shape[1]])
    # plt.show()

    return eval, Y, clf_rfc, importances, indices
    

#%%
_verbose = False
_pool = None  # Pool()
_manager = None  # Manager()
change_work_dir()
valid_ids, data_dict = load_files()
count_id_occurence()
var_tags = build_variable_tag_set()
to_drop =\
    var_tags["tx"].union(var_tags["sp"]).union(var_tags["wa"]).union(var_tags["su"]).union(var_tags["di"]).union(var_tags["Ve"])
keys = valid_ids
dd_tmp = load_pickle("../b_Preprocessing/Data/data_dict_tmp")


# %%
# check which negative values exist
# check percentage of columns being negative
lst = []
for df_k in dd_tmp.keys():
    df = dd_tmp[df_k]
    neg_val = (df < 0).sum().sum()
    neg_cols = (df < 0).sum()[(df < 0).sum() > 0].index
    avg_no_neg_vals = neg_val / len(neg_cols)
    lst.append((df_k, neg_val, neg_val / (df.shape[0] * df.shape[1]), avg_no_neg_vals))

lst.sort(key=lambda x: x[1], reverse=True)
print_table(["DataFrame", "# neg. values", r"% of values", "Avg. neg. Values"], lst)


#%% 
id_counts = keys["ID_t"].value_counts()
ids = keys[keys["ID_t"].isin(id_counts[id_counts > 1].index)]
# ids = keys[keys["ID_t"].isin(id_counts[id_counts == 1].index)]
df = data_dict["spVocTrain"]
df2 = pd.merge(df, ids, on=["ID_t", "spell"], how="inner")
last_id = -1
i = 1
spell_rdx = []
for _id in df2["ID_t"]:
    if _id == last_id:
        i += 1
    else:
        i = 1
    spell_rdx.append(i)
    last_id = _id

df2["spell_redux"] = spell_rdx


#%%
succ = df2[df2.ts15218 == 1]["spell_redux"]
drop = df2[df2.ts15218 == 2]["spell_redux"]
oth = df2[df2.ts15218 < 1]["spell_redux"]
print(succ.shape, drop.shape, oth.shape)

fig, ax = plt.subplots(1, figsize=(12,8))
_ = ax.hist(
    [succ, drop, oth],
    # bins = 4,
    # density=True,
    stacked=True,
    # cumulative=True,
    #rwidth=.8,
    label=["Success", "Dropout", "Other"]
)
ax.legend()

#%%
df = keys.merge(data_dict["spVocTrain"][["ID_t", "spell", "ts15218"]], on=["ID_t", "spell"], how="inner")
print(df.shape, df[df["ts15218"] == 1].shape, df[df["ts15218"] == 2].shape, df[~df["ts15218"].isin([1,2])].shape)
df[df["ts15218"] < 1] = np.nan
df = df.dropna(subset=["ts15218"])
print(df.shape)
df = df.sort_values(by=["ID_t", "spell", "ts15218"], axis="index").drop_duplicates(subset=["ID_t"])
# df = keys.sort_values(by=["ID_t", "spell"], axis="index").drop_duplicates(subset=["ID_t"])
print(df.shape, df[df["ts15218"] == 1].shape, df[df["ts15218"] == 2].shape, df[~df["ts15218"].isin([1,2])].shape)
unique_keys = df.drop(labels="ts15218", axis="columns")

#%% [markdown]
# There is only 1 id with 4 entries in vocTrain, and 25 with 3 entries. There are 577 with 2 entries. Leaving 7457 ids with only a single entry.
#
# If I cut out all those with more than 1 educational B.Sc. spell, I lose 1114 successfull studies (ugh, overperformers) and 79 drop outs. I could reduce that loss by only throwing away only one of the spell, instead of the entire student. I could also priorites drop out spells over successfull spells.

#%%
from datetime import date
## Still need to fix cati having too many missing / negative coded values before throwing out rows
#### Either use fillna wih ffill and bfill or find a smarter way, maybe something more content aware?
# lambda df: df.fillna(axis="columns", method="ffill").fillna(axis="columns", method="bfill"

def a(df):
    # enddate = date(year=df["year"].iloc[0], month=df["month"].iloc[0], day=1)

    # df["wave_date"] = df["wave"].apply(lambda x: date(year=waveid2year[x], month=1, day=1))
    # df_filtered = df[(df["wave_date"] <= enddate) | (df["wave"] == 1)]
    # df_filtered.drop(labels="wave_date", axis="columns", inplace=True)
    # df.drop(labels="wave_date", axis="columns", inplace=True)

    # return df_filtered
    return df[
            (df["wave"].apply(lambda x: date(year=waveid2year[x], month=1, day=1)) <= date(year=df["year"].iloc[0], month=df["month"].iloc[0], day=1))
            | (df["wave"] == 1)
        ]


vocTrain = data_dict["spVocTrain"]
merged = unique_keys.merge(vocTrain, on=["ID_t", "spell"], how="inner")

spellend = merged[["ID_t"] + vocTrain_spellend]
spellend.columns = ["ID_t", "month", "year"]

cati = data_dict["pTargetCATI"]
cati = cati[cati["ID_t"].isin(merged["ID_t"])]
cati = cati.merge(spellend, on="ID_t", how="left")
cati.groupby("ID_t").apply(a)

## This is where we then boil down the filtered cati version to a one-vector-per-id dataframe, so we can properly join
## So we need to find out, how much info actually is in any row, find differences, etc.
## A lot of difference might come from unique values only asked in one interview.
## If we copy that down to all rows however, that information difference might be small
## Then we "only" need to find a way to merge / decide and result in one row
##
## This procedure should be applicable to cawi as well, but we will see.

#%%
if False:
    vocTrain = data_dict["spVocTrain"]
    merged = df.drop(axis=1, labels="ts15218").merge(vocTrain, on=["ID_t", "spell"], how="inner")
    coded_nan = [
        -94, # not reached
        -95, # implausible value
        -98, # don't know
        -54, # missing by design
        -90, # unspecified missing
        -93, # does not apply
        -99, # filtered
        -52, # implausible value removed
        -53, # anonymized
        -55, # not determinable
        -56, # not participated
    ]
    interesting_coded = [
        -97 # refused
    ]
    interesting_coded += list(range(-29, -19)) # various

    merged[merged.isin(coded_nan)] = np.nan
    merged = merged.dropna(axis="columns", how="all")
    # merged[(merged < 0) & merged.notna()].dropna(how="all").dropna(axis="columns", how="all")
    nacount = ((merged.isna()) | (merged < 0)).sum()
    drop = nacount[nacount > merged.shape[0] / 2].index.to_list() # drop columns with more than half na
    # merged = merged.drop(labels=drop, axis="columns")
    merged.set_index("ID_t", inplace=True)
    Y = merged.pop("ts15218").values
    merged[(merged.isna()) | (merged < 0)] = -1

    #%%
    ohec = OneHotEncoder(sparse=False, categories="auto")
    transformed = ohec.fit_transform(merged)
    xTrain, xTest, yTrain, yTest = train_test_split(transformed, Y, test_size=0.2)#, random_state=4)
    clf = SVC()
    clf = clf.fit(xTrain, yTrain)

#%%
if False:
    # feat_importance = dtc.tree_.compute_feature_importances(normalize=False)
    # feat_importance[np.where(feat_importance > 0)]
    clf.score(xTest, yTest)

#%%
if None:
    import graphviz
    from sklearn import tree
    dot_data = tree.export_graphviz(dtc, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("test")

#%%
if False:
    df = data_dict["spSibling"].copy()
    df.fillna(-420, inplace=True)
    
    ohec = OneHotEncoder(sparse=False, categories='auto')
    pca = PCA(n_components=40)
    transformed = ohec.fit_transform(df[df.columns[1:]])
    dummies = pd.get_dummies(df, columns=df.columns[1:])

    components = pca.fit_transform(transformed)
    pd.DataFrame(data=components)
    pca.explained_variance_ratio_.sum()

#%% [markdown]
# ### Joining Information and Considering Dropped Cases
# Since the first wave information on intended degree is not in spVocTrain, but only in pCATI, we will join that one column into our dataframe.
# 
# #### Dropped Students
# NEPS considers a change of university dropping out, which I do not. This will needed to be filtered for. I also need to find a way to clean and rebuild the data in a concise one-row format, instead of the episodes being spread out over multiple.
# 
# ##### Figuring out
# Things to look out for:
# * Weird disagreeing spells
# * No sucessfull graduation (2)
#     * With no spell afterwards
#     * With a spell afterwards, but a change of subject
# 
# ##### To be considered
# Dropped out, if ts15218 "succesfull graduation" has a value of 2 "no", spells of interest are students doing some bachelor degree at a university "Universität" or school of applied science "Fachhochschule".
# 
# Cases:
# * dropped out -> no further spell of interest ✓
# * dropped out -> spell of interest
#     * -> changed subject, irrelevant of change of institution ✓
#     * -> changed institution, but not subject ☓
