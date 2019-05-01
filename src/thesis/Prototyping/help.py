import numpy as np
import pandas as pd
import os
import os.path


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


def get_valid_ids(fn="valid_ids.txt"):
    with open(fn, "r", encoding="utf-8") as fh:
        valid_ids = fh.read().split(",")
    valid_ids = [int(id_) for id_ in valid_ids]
    return np.array(valid_ids)


def build_data_dict(valid_ids):
    files = os.listdir("../a_RawData")
    rawdatafiles = []
    for fn in files:
        if fn.endswith(".dta") and fn.startswith(("SC5_sp", "SC5_p")):
            rawdatafiles += [fn]

    data_dict = {}
    for df in rawdatafiles:
        name = df.split("_")[1]
        print(f"Loading {name}...", end=" ", flush=True)
        data_dict[name] = pd.read_stata(
            "../a_RawData/" + df, convert_categoricals=False)
        data_dict[name] = data_dict[name][data_dict[name].ID_t.isin(valid_ids)]
        print(f"Done.", flush=True)

    #%%
    print(f"{'Total IDs':<20}{valid_ids.shape[0]}\n")
    for df_key in data_dict.keys():
        df = data_dict[df_key]
        print(f"{df_key:<20}", end="")
        print(f"{df.ID_t.unique().shape[0]}")

    return data_dict


def filter_df(df_key, data_dict, valid_ids, verbose=False):
    if verbose:
        print(f"Now checking {df_key}...", flush=True)
    df = data_dict[df_key]
    df = df[df["ID_t"].isin(valid_ids)]

    if verbose:
        print(f">{df_key:<15}: Filter 1, dropna...", flush=True)
    cols_to_keep = df.dropna(axis=1, how="all").columns
    if verbose:
        print(f">{df_key:<15}: Filter 2, not 1 per id...", flush=True)
    tmp = df.notna().sum() >= df["ID_t"].unique().shape[0]
    cols_to_keep = np.union1d(cols_to_keep, tmp[tmp == True].index)
    if verbose:
        print(f">{df_key:<15}: Filter 3, groupby...", flush=True)
    groupcol = "ID_t"
    tmp = df.groupby(groupcol).agg(lambda x: (
        x.first_valid_index() == x.last_valid_index()) and x.notna().any()).all()
    tmp = tmp[tmp == True]
    cols_to_keep = np.union1d(cols_to_keep, tmp.index)

    valid_cols = len(cols_to_keep)
    rem_cols = len(df.columns) - valid_cols
    contains = len(df["ID_t"].unique())

    if verbose:
        print(f">{df_key:<15}: Done...", flush=True)
    tpl = (df_key, contains, contains /
            len(valid_ids)*100, valid_cols, rem_cols)
    return tpl


def filter_data(data_dict, valid_ids):
    from multiprocessing import Pool as ThreadPool, Manager
    from functools import partial

    pool = ThreadPool(8)
    func = partial(filter_df, verbose=True, data_dict=data_dict, valid_ids=valid_ids)

    ls = list(pool.imap(func, list(data_dict.keys())))
    ls.sort(key=lambda x: x[1], reverse=True)
    print(f"There are {len(valid_ids)} unique students considered.\n")
    print_table(["Filename", "Contains", "Percentage", "Valid Cols", "Removed cols"], ls, max_width=30)


def main():
    print("Loading ids...", end="")
    valid_ids = get_valid_ids()
    print("Done.")
    print("Building data dict...")
    data_dict = build_data_dict(valid_ids)
    print("Filtering...")
    filter_data(data_dict, valid_ids)

if __name__ == "__main__":
    main()
