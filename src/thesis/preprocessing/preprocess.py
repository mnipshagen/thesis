"""
Run the preprocess pipeline. Load files, drop files according to criterion, clean files, aggregate files, join files.
"""
import os, sys
import numpy as np
import pandas as pd
from collections import Counter
from functools import partial
from multiprocessing import Pool, Manager

from . import aggregate
from ..util import print_table, save_pickle, load_pickle

_verbose = False
_pool = None
_manager = None


def build_variable_tag_set(df):
    """
    Builds a set of tags with the first two characters of the variable name.

    The first two characters of the variables in the NEPS data set are identifiers for the kind of variable.
    """
    d = dict()
    cols = df.columns
    for col in cols:
        tag = col[:2]
        if tag in d:
            d[tag].add(col)
        else:
            d[tag] = {col}

    return d


def mp_load_dta(filename, data_dict, valid_ids, drop_df_if_students_less_than):
    """
    Reads in the given dta file, but only if there are at least `drop_df_if_students_less_than` ids of the total ids in the file
    """
    name = os.path.basename(filename).split("_")[1]
    # don't convert categoricals, since that has lead to issues
    df = pd.read_stata(filename, convert_categoricals=False)
    # check if there are enough ids
    if df["ID_t"].unique().shape[0] < (valid_ids.shape[0] * drop_df_if_students_less_than):
        print(f"{name} was dropped due to not having enough ids. Only had {df['ID_t'].unique().shape[0]/valid_ids.shape[0]*100:.2f}%.")
        return
    # only include numeric columns, no free text. It is mostly redacted anyway.
    df = df.select_dtypes(include=np.number)
    # only keep students that have spells we are interested in
    df = df[df["ID_t"].isin(valid_ids["ID_t"])]
    # if the file has subspells, only keep complete ones
    try:
        df = df[df["subspell"] == 0]
    except KeyError:
        pass

    # give back the name of the file and the dataframe
    data_dict[name] = df
    return name, df


def create_valid_ids(valid_ids_path="./data/interim/valid_ids"):
    """
    Filter the dataset for the id's we are interested in and return a dataframe with each id-spell combination that is of interest
    """
    vocTrain = pd.read_stata("./data/raw/SC5_spVocTrain_D_11-0-0.dta", convert_categoricals=False)
    CATI = pd.read_stata("./data/raw/SC5_pTargetCATI_D_11-0-0.dta", convert_categoricals=False)

    variables = [
        "ID_t",  # target id
        "wave",  # wave of episode
        "spell",  # spell id
        "ts15221_v1",  # angestrebter Ausbildungsabschluss
        "ts15219_v1",  # Ausbildungsabschluss: B.Sc, M.Sc, Ausbildung, etc.
        # "ts15265",  # Note des Abschluss
        # "tg24103",  # Episodenmodus
        "ts15201",  # Ausbildungstyp
        "ts15218",  # Erfolgreicher Abschluss
        "tg24159",  # Fachwechsel gegenüber Vorepisode; 1=same, 2=diff
        "tg24121",  # Hochschulwechsel gegenüber Vorepisode; 1=same, 2=diff
        # "ts1512c",  # Andauern der Episode
        # "ts1511m_g1",  # Prüfmodul: Startdatum (Monat, ediert)
        # "ts1511y_g1",  # Prüfmodul: Startdatum (Jahr, ediert)
        # "ts1512m_g1",  # Prüfmodul: Enddatum (Monat, ediert)
        # "ts1512y_g1",  # Prüfmodul: Enddatum (Jahr, ediert)
    ]

    # drop incomplete spells
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
        subset["ts15221_v1"].isin(relevant_courses_of_study)  # filter bachelor only
    ]

    # prepare subset of CATI for joining
    other = CATI[
        # the question was only asked in wave 1, and this way ID_t becomes the unique identifier
        CATI["wave"] == 1
        & CATI["tg02001"].isin([1, 3])  # filter for 3=BA and 1=BA Lehramt only
    ][["tg02001", "ID_t"]]  # ID_t to join on, tg02001 holds the pursued degree

    # join CATI subset into dataframe for more complete sample
    curr = curr.join(other.set_index("ID_t"), on="ID_t")  # integrate tg02001 into df to fill in ts15201 -28
    # filter out those rows that CATI added nothing too, meaning those without information on pursued degree even after joining
    curr = curr[(curr["ts15201"] != -28) |
                ((curr["ts15201"] == -28) & (~curr["tg02001"].isnull()))]

    # reorder columns in tmp dataframe for a nicer overview, moving joined CATI degree next to vocTrain degree
    cols = list(curr.columns)
    purs_deg_idx = cols.index("ts15221_v1") + 1
    cols = cols[:purs_deg_idx] + cols[-1:] + cols[purs_deg_idx:-1]
    curr = curr[cols]

    # retrieve all unsuccesful episodes
    dropped = curr[(curr["ts15218"] == 2)]

    # join into unique key combination
    valid_ids = dropped[["ID_t", "spell"]].append(curr[curr["ts15218"] == 1][["ID_t", "spell"]])

    valid_ids["tmp"] = valid_ids["ID_t"].astype(int).astype(str) + "_"  + valid_ids["spell"].astype(int).astype(str)
    valid_ids = valid_ids.drop_duplicates("tmp").drop(columns="tmp")

    if valid_ids_path:
        save_pickle(valid_ids, valid_ids_path)

    return valid_ids


def build_data_dict(
    valid_ids,
    drop_df_if_students_less_than=0.45,
    data_path="./data/raw/",
    data_dict_path="./data/interim/data_dict"
    ):
    """
    Load all data files and put them into a dictionary for easier access.

    Runs in parallel if available.
    """
    # Get all spell and panel dta files from the directory
    files = os.listdir(data_path)
    rawdatafiles = []
    for fn in files:
        if fn.endswith(".dta") and fn.startswith(("SC5_sp", "SC5_p")):
            rawdatafiles.append(os.path.join(data_path, fn))
    print(f"Found following data files: {rawdatafiles}")

    # run in parallel
    if _pool and _manager:
        data_dict = _manager.dict()
        func = partial(
            mp_load_dta,
            data_dict=data_dict, 
            valid_ids=valid_ids,
            drop_df_if_students_less_than=drop_df_if_students_less_than
        )
        # load files
        list(_pool.map(func, rawdatafiles))
    else:
        # load 1 by 1
        data_dict = dict()
        for fn in rawdatafiles:
            _name, df = mp_load_dta(fn, data_dict, valid_ids, drop_df_if_students_less_than)

    if _verbose:
        print(f"{'Total IDs':<20}{valid_ids.shape[0]}\n")
        for df_key in data_dict.keys():
            df = data_dict[df_key]
            print(f"{df_key:<20}{df['ID_t'].unique().shape[0]}\t{df.select_dtypes(exclude=np.number).columns}")
    
    data_dict = dict(data_dict)
    if data_dict_path:
        save_pickle(data_dict, data_dict_path)

    return data_dict


def mp_copy_missing(df):
    """
    Replace missing values that seem to hold no value with NAN and then backwards and forwards fill those columns in which only one cell holds a proper value. The assumption is, that those are questions only asked once and never again.
    """
    df[df.isin([-54, -90, -93])] = np.nan
    sum_na = (df.notna() & (df >= 0)).sum()
    fill_cols = sum_na[sum_na == 1].index
    df[fill_cols] = df[fill_cols].ffill()
    df[fill_cols] = df[fill_cols].bfill()
    # variables that will be replaced, either due to insignificance or underrepresentation
    coded_nan = [
        -94,  # not reached
        -95,  # implausible value
        -98,  # don't know
        -54,  # missing by design
        -90,  # unspecified missing
        -93,  # does not apply
        -99,  # filtered
        -52,  # implausible value removed
        -53,  # anonymized
        -55,  # not determinable
        -56,  # not participated
    ]
    # variables that will be kept
    interesting_coded = [
        -97 # refused
    ]
    interesting_coded += list(range(-29, -19)) # various
    # replace
    df[df.isin(coded_nan)] = np.nan

    return df


def copy_missing(data_dict):
    """
    For each dataframe, fill the missing values of columns which hold only 1 value per id.
    """
    for df_k in data_dict.keys():
        if not ("CATI" in df_k or "CAWI" in df_k):
            continue
        df = data_dict[df_k]
        if _verbose:
            print(f"Cleaning {df_k} with shape {df.shape}....", end="", flush=True)
        grp = df.groupby(["ID_t"])
        clean_grps = _pool.map(
            mp_copy_missing,
            [group for name, group in grp]
        )
        data_dict[df_k] = pd.concat(clean_grps)
        if _verbose:
            print(f"Done! Output has shape {data_dict[df_k].shape}", flush=True)
    
    return data_dict


def drop_duplicate_vars_deprecated(data_dict):
    """
    Deprecated: This was basically to print out the amount of versionated and generated variables in each dataset.
    """
    table_cols = ["Column", "has v", "has g", "na_missing_v", "na_missing_g", "na_missing_non", "has other"]
    lst = []
    for df_k in data_dict.keys():
        lst = []
        print(f"{df_k}\n{'':#<40}")
        df = data_dict[df_k]
        cols = df.columns
        column_set_dict = dict()
        for col in cols:
            # if "_v" in col:
            #     info[1] += 1
            # if "_g" in col:
            #     info[2] += 1
            col_name = col.split("_")[0]
            if col_name in column_set_dict.keys():
                column_set_dict[col_name] = column_set_dict[col_name].union({col})
            else:
                column_set_dict[col_name] = {col}
        for col_name in column_set_dict.keys():
            set_ = column_set_dict[col_name]
            if len(set_) == 1:
                continue
            info = [col_name, "False", "False", 0, 0, 0, "False"]
            for col in set_:
                if col.endswith(("R", "O")):
                    continue
                idx = -1
                if "_v" in col:
                    info[1] = "True"
                    idx = 3
                elif "_g" in col:
                    info[2] = "True"
                    idx = 4
                elif not "_" in col:
                    idx = 5
                elif "_w" in col:
                    info[6] = "True"
                    continue
                else:
                    info[6] = "True"
                    print(f"{col} fits no scheme")
                if idx != -1: info[idx] += (df[col].isna() | (df[col] < 0)).sum()
        
            if info[1] == "True" or info[2] == "True": lst.append(info)
        if len(lst) > 0: print_table(table_cols, lst)
    return data_dict


def drop_duplicate_vars(data_dict):
    """
    Remove duplicate variables. Keep version with least nan

    For all variables, only retain one version if there are duplicates.
    """
    ### if var is time variable (ends in m,y) always keep g1 var
    ### there might be variables with only a g version
    ### there are some with several v/g versions
    ### if ends in "a" use normal if available
    ## always throw out O and R vars, they are unavailable in the download version
    ## v > g?
    for df_k in data_dict:
        # print(f"{'':=<30}\n{'  '+df_k+'  ':=^30}\n{'':=<30}")
        print(f"{'  '+df_k+'  ':=^30}")
        df = data_dict[df_k]
        cols_to_drop = []
        cols = df.columns
        print(f"Has {len(cols)} columns")
        column_dict = {}
        for c in cols:
            cb = c.split("_")[0]
            if cb in column_dict:
                column_dict[cb].append(c)
            else:
                column_dict[cb] = [c]
        # get base names to identify duplicates
        double_cols = {base: cols for base, cols in column_dict.items() if len(cols) > 1}
        for col_base in double_cols.keys():
            cols = double_cols[col_base]
            # ignore wide variables
            if all(["_w" in c for c in cols]):
                continue
            # drop redacted
            drop = [c for c in cols if c.endswith(("O", "R"))]
            if len(cols) - len(drop) in [0,1]:
                cols_to_drop += drop
                continue
            # special rule for time variables
            if col_base.endswith(("m", "y")) and all([s in double_cols.keys() for s in [col_base[:-1] + "m", col_base[:-1] + "y" ]]):
                if any(["_g" in c for c in cols]):
                    drop += [c for c in cols if not c[:-1].endswith("_g")]
            # vars ending with a and c seem to be outliers in their behaviour
            if col_base.endswith("a") and any([not "_" in c for c in cols]):
                drop += [c for c in cols if "_" in c or not c.endswith("a")]
            if col_base.endswith("c"):
                drop += [c for c in cols if not "_" in c ]
            # just keep the variables with the least nan
            if len(cols) - len(drop) > 1:
                nas = []
                for c in cols:
                    nas.append(float("inf") if c in drop else ((df[c] < 0) | (df[c].isna())).sum())
                min_ = nas.index(min(nas))
                drop += cols[:min_] + cols[min_+1:]
            
            drop = list(set(drop))
            if len(cols) - len(drop) != 1:
                print(f"!!! There is not exactly 1 column left !!!\nHad  : {sorted(cols)}\nDrop : {sorted(drop)}")
            cols_to_drop += drop

        cols_to_drop = list(set(cols_to_drop))
        # does not return due to side effects
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped {len(cols_to_drop)} columns")


def join(valid_ids, data_dict, agg_dict, voctrain=True, original=False):
    # unique keys identifying a row in each file
    # df_unique_keys = {
    #     "pTargetCATI": ["ID_t", "wave"],
    #     "pTargetCAWI": ["ID_t", "wave"],
    #     "spChild": ["ID_t", "child", "subspell", ["wave"]],
    #     "spChildCohab": ["ID_t", "spell", "subspell", ["child", "wave"]],
    #     "spCourses": ["ID_t", "wave", "splink", ["sptype", "course_w1", "course_w2", "course_w3"]],
    #     "spEmp": ["ID_t", "spell", "subspell", ["wave"]],
    #     "spFurtherEdu1": ["ID_t", "course", ["wave"]],
    #     "spFurtherEdu2": ["ID_t", "course", ["wave"]],
    #     "spGap": ["ID_t", "spell", "subspell", ["wave" "splink"]],
    #     "spInternship": ["ID_t", "spell", "subspell", ["wave" "splink"]],
    #     "spMilitary": ["ID_t", "spell", "subspell", ["wave" "splink"]],
    #     "spParLeave": ["ID_t", "spell", "subspell", ["wave" "child" "splink"]],
    #     "spPartner": ["ID_t", "partner", "subspell", ["wave"]],
    #     "spSchool": ["ID_t", "spell", "subspell", ["wave" "splink"]],
    #     "spSchoolExtExam": ["ID_t", "exam", ["wave"]],
    #     "spSibling": ["ID_t", "sibling", ["wave"]],
    #     "spUnemp": ["ID_t", "spell", "subspell", ["wave" "splink"]],
    #     "spVocExtExam": ["ID_t", "exam", ["wave"]],
    #     "spVocTrain": ["ID_t", "spell", "subspell", ["wave" "splink"]]
    # }
    if _verbose:
        print("Starting join...")
    to_be_merged = data_dict["spVocTrain"]
    # case of excluding voctrain
    if not voctrain:
        to_be_merged = to_be_merged[["ID_t", "spell", "ts15218"]]
    joined = pd.merge(valid_ids, to_be_merged, on=["ID_t", "spell"], how="inner")
    for dfk in agg_dict.keys():
        df = agg_dict[dfk]
        if _verbose:
            print(f"Joining {dfk}...")
            print(f"Shape of joined before: {joined.shape}...")
            print(f"Shape of {dfk}: {df.shape}")

        joined = pd.merge(joined, df, on="ID_t", how="left")

        if _verbose:
            print(f"New shape of joined: {joined.shape}\n")

    print("Some final cleanup..")
    cols = joined.shape[1]
    joined.dropna(axis="columns", how="all", inplace=True)
    print(f"Dropped {cols-joined.shape[1]} null columns.")
    na = joined.isna().sum().sum()
    joined.fillna(-1, inplace=True)
    print(f"Replaced {na:,} nan values with -1.")
    cols = joined.shape[1]
    thresh = 0.75
    cols_to_drop = []
    for c in joined.columns:
        if (joined[c] < 0).sum() > thresh * joined.shape[0]:
            cols_to_drop.append(c)
    # joined.drop(columns=cols_to_drop, inplace=True)
    var_tags = build_variable_tag_set(joined)
    cols_to_drop = set(cols_to_drop)
    for k in ["ID", "tx", "sp", "wa", "su", "di", "Ve", "h_"]:
        try:
            cols_to_drop = cols_to_drop | var_tags[k]
        except KeyError:
            pass
    # the rules applied with correction 1, 2 and 3
    if not original:
        cols_to_drop = cols_to_drop | set(["ts15265", "ts15219_v1", "tg50007"]) | set([c for c in joined.columns if c.split("_")[0].endswith("y")])
    cols_to_drop = list(cols_to_drop)
    joined.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    print(f"Dropped {cols-joined.shape[1]} columns which had more than {thresh*100}% missing values.")
    print(f"Final shape of output: {joined.shape[0]} x {joined.shape[1]}")

    return joined


def main(
        redo=False,
        valid_ids_path="./data/interim/valid_ids",
        data_dict_path="./data/interim/data_dict",
        agg_data_dict_path="./data/interim/data_dict_agg",
        output_df_path="./data/out/data",
        include_voctrain=True,
        original_data=False,
        multithreading=-1, 
        verbose=True
    ):
    """
    Run the whole pipeline: Build valid ids -> build data dict -> filter data dict -> fill data dict -> aggregate data dict -> join
    """
    paths = ["./data", "./data/interim", "./data/out"]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    global _verbose, _pool, _manager
    _verbose = verbose
    if multithreading:
        _pool = Pool() if multithreading == -1 else Pool(multithreading)
        _manager = Manager()
    
    valid_ids = data_dict = agg_data_dict = joined_frame = None
    if not redo:
        valid_ids = load_pickle(valid_ids_path)
        data_dict = load_pickle(data_dict_path)
        agg_data_dict = load_pickle(agg_data_dict_path)
        joined_frame = load_pickle(output_df_path)

    if redo or any([x is None for x in [valid_ids, data_dict, agg_data_dict, joined_frame]]):
        if valid_ids is None:
            valid_ids = create_valid_ids(valid_ids_path=valid_ids_path)
        if data_dict is None:
            data_dict = build_data_dict(valid_ids, data_dict_path=data_dict_path)
            data_dict = dict(copy_missing(data_dict))
            drop_duplicate_vars(data_dict)
            save_pickle(data_dict, data_dict_path)
        if agg_data_dict is None:
            agg_data_dict = aggregate.agg(valid_ids, data_dict, mp_pool=_pool, mp_manager=_manager, verbose=_verbose)
            save_pickle(agg_data_dict, agg_data_dict_path)
        if joined_frame is None:
            joined_frame = join(valid_ids, data_dict, agg_data_dict, voctrain=include_voctrain, original=original_data)
            save_pickle(joined_frame, output_df_path)

    return joined_frame


if __name__ == "__main__":
    try:
        os.chdir(os.path.join(os.getcwd(), 'src/thesis'))
        print("cwd:", os.getcwd())
    except Exception as e:
        print(e)
        pass
    main(redo=False)


{ #notes on aggregation. Kept for reference.
    # aggregate spell data
    # sibilings : amount of siblings, youngest=0; middle=1; oldest=2, do we do something with sibling vocation, education?
    # military  : type of service (including none)
    #               Amount of military episodes
    #               
    # gap       : type of gap (or none)
    #               Amount of gaps
    #               Training programs during gap: ts29201
    #               dauer: ts2912m|ts2912y - ts2911m|ts2911y
    #               Type of gap episode: ts29101
    # unemploy  : amount of unemployment?
    #               Amount
    #               Dauer: ts2512m|ts2512y - ts2511m|ts2511y
    #               Receipt of unemployment benefits or support at the beginning: ts25202
    #               currently registered as unemployed: ts25203
    #               Number of job applications: ts25205
    #               Invitation to job interviews: ts25206
    #               Number of interviews: ts25207
    #               Par􀆟cipa􀆟on in programs for further educa􀆟on financed by employment agency: ts25209
    # intern    : amount of internships, avg pay, avg working hours, (freiwillig, pflicht, mix)
    # school    : ?? this is a lot
    # partner   : I am not sure, maybe #partners, cohabitation, marriage
    #               max(partner)
    #               Living together with partner: t733030
    #               End of the partnership: ts31510
    #               Current school partner: tg28320
    #               Current professional education partner: tg28321 & 
    #               Currently attended type of university partner: tg28323
    #               Marriage / registered civil partnership: ts31410
    #               Highest general school-leaving qualifica􀆟on of partner: ts31212
    #               highest vocational qualification partner: ts31214
    #               Partner: Place of residence: t407021
    #               Frequency of contact with partner: t733005
    # emp       : a lot of jobs done, avg pay, hours etc.
    #               amount
    #               Job Description: ts23201
    #               While Studying: tg2608a
    #               Type of Student Employment: tg2608b
    #               Professional Position: ts23203 &
    #               Exact Professional Position: ts23204
    #               duration: ts2312m|ts2312y - ts2311m|ts2311y

    ###################
    # In final consideration I left out all variables indicating "year" as this would decrease the models generalisation ability. If students from 20 years ago should be evaluated, the model should still work the same. Since values are treated as nominal, the year 2018 is in no relation to the year 1996.
    # sometimes we just choose the first value of the bunch... I just have no good clue to aggregate categories
    ###################
    # Random Forest 5000 Trees evaluation
    # spSibling             Forest scored: 0.7554
    # tg3270y   : 0.3604    year of sibling’s birth
    # tg3270m   : 0.2766    Month of sibling’s birth
    # tg32711   : 0.1062    Sibling’s highest school-leaving qualification

    # spGap                 Forest scored: 0.7809
    # ts2912m_g1: 0.2144    Check module: Ending date (month, edited)
    # ts2911m_g1: 0.2082    Check module: Starting date (month, edited)
    # ts2911y_g1: 0.1299    Check module: Starting date (year, edited)
    # ts2912y_g1: 0.1105    Check module: Ending date (year, edited)
    # ts29101   : 0.0903    Type of gap episode

    # spMilitary            Forest scored: 0.7548
    # ts2112m_g1: 0.2428    Biography: Ending date of spell (month, edited)
    # ts2111m_g1: 0.2145    Biography: Starting date of spell (month, edited)
    # ts2111y_g1: 0.1575    Biography: Starting date of spell (year, edited)
    # ts2112y_g1: 0.1347    Biography: Ending date of spell (year, edited)
    # ts21201   : 0.0891    Type of military service episode
    # ts21202   : 0.0391    Attendance of seminars/training courses during military service

    # spUnemp               Forest scored: 0.8106
    # ts2511m_g1: 0.1484    Biography: Starting date of spell (month, edited)
    # ts2512m_g1: 0.1336    Biography: Ending date of spell (month, edited)
    # ts2511y_g1: 0.0850    Biography: Starting date of spell (year, edited)
    # ts25205   : 0.0745    Number of job applications
    # ts2512y_g1: 0.0737    Biography: Ending date of spell (year, edited)
    # ts25901   : 0.0686    Auxiliary variable current unemployment
    # ts25207   : 0.0562    Number of interviews

    # spInternship          Forest scored: 0.8777
    # tg3607m_g1: 0.0822    Biography: Starting date of spell (month, edited)
    # tg3608m_g1: 0.0801    Biography: Ending date of spell (month, edited)
    # tg36114   : 0.0625    Amount of remuneration for the internship
    # tg36111   : 0.0597    Average working time in internship
    # tg36110   : 0.0389    Type of internship
    # tg3607y_g1: 0.0362    Biography: Starting date of spell (year, edited)
    # tg36118   : 0.0361    Relation of internship to degree course
    # tg3608y_g1: 0.0354    Biography: Ending date of spell (year, edited).
    # t265323   : 0.0310    Learning content: Autonomy 3
    # t265321   : 0.0301    Learning content: Autonomy 1.
    # t265302   : 0.0299    Learning content: Comprehensiveness 2

    # spSchool              Forest scored: 0.8196
    # ts11218   : 0.0907    Overall grade on final certificate
    # t724804   : 0.0756    4th ’Abitur’ subject
    # t724802   : 0.0637    2nd ’Abitur’ subject
    # t724803   : 0.0611    3rd ’Abitur’ subject
    # ts1111y_g1: 0.0591    Biography: Starting date of spell (year, edited)
    # t724801   : 0.0585    1st ’Abitur’ subject
    # ts1112y_g1: 0.0544    Biography: Ending date of spell (year, edited)
    # t724712   : 0.0528    Final semester points in mathematics
    # t724805   : 0.0485    5th ’Abitur’ subject
    # t724714   : 0.0468    Points earned in last semester of German

    # spPartner             Forest scored:0.8359
    # tg2811m   : 0.0610    Start date Partnership Month
    # ts3120y_v1: 0.0600    Partner’s year of birth
    # tg2811y   : 0.0481    Start date Partnership Year
    # tg2804m   : 0.0476    End date Partnership episode (month)
    # ts31226_g2: 0.0414    Partner: Occupation (KldB 2010)
    # ts31224_v1: 0.0362    Working hours, partner.
    # ts31212_g1: 0.0318    Partner: Highest educational achievement (ISCED-97)
    # t733005   : 0.0309    Frequency of contact with partner
    # tg2812m   : 0.0295    Date when couple moved in together (month)
    # ts31214_v1: 0.0290    Partner: highest professional qualifica􀆟on

    # spEmp                 Forest scored:0.8860
    # tg23228   : 0.0460    Relation to studies of non-student employment
    # ts23201_g2: 0.0310    Job description (KldB 2010)
    # ts23228   : 0.0301    Type of education required
    # ts23223   : 0.0297    Actual weekly working hours at the moment/at the end
    # ts23510   : 0.0288    Gross income, open
    # ts23410   : 0.0285    Net income, open
    # ts2311m_g1: 0.0246    Check module: Starting date (month, edited)
    # ts23240_g1: 0.0242    Economic sector (WZ 2008)
    # ts23219   : 0.0237    Contractual/actual work hours at start of job

    # pTargetCAWI           Forest scored:0.9859
    # tg50007   : 0.0494
    # tg51004   : 0.0144
    # t29175f   : 0.0130
    # t29187b   : 0.0110
    # t29175c   : 0.0093
    # tg51311_g2: 0.0093
    # t29187a   : 0.0081
    # tg51312   : 0.0075
    # t29170f   : 0.0072
    # tg52011   : 0.0071
    # t29173a   : 0.0069
    # t291800   : 0.0068
    # t29187c   : 0.0068
    # t29171b   : 0.0067
    # t29170d   : 0.0064

    # pTargetCATI           Forest scored: 0.9969
    # tg2412g   : 0.0159
    # tg2412e   : 0.0154
    # tg2412o   : 0.0153
    # tg2412v   : 0.0152
    # tg2412r   : 0.0151
    # tg2412x   : 0.0151
    # tg2412d   : 0.0149
    # tg2412s   : 0.0149
    # tg2412p   : 0.0148
    # tg2412m   : 0.0146
    # tg2412n   : 0.0144
    # tg2412i   : 0.0143
    # tg2412h   : 0.0139
    # tg2412l   : 0.0139
    # tg2412q   : 0.0129

    
    # vocTrain  : this is the base line. no aggregation
    # CAWI      : drop all waves after voc spell, theeeeen append? this will lead to inconsistent vector lengths.. only keep first as a base line? We could do the same for the others including wave information.
    # CATI      : see above
    # join(valid_ids, data_dict)
}