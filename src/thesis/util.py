import pickle, sys, os

import numpy as np
import joblib


def print_table(cols, data, max_width=None, prefix=None, suffix=None):
    """
    Utility function to print nice tables given a list of headers and a list of tuples
    """
    # set column width to given max_width or max width of column entry
    widths = [max_width if max_width else 1e309 for _ in cols]
    for idx, width in enumerate(widths):
        def my_len(x): return len(f"{x:.3f}" if isinstance(
            x, (float, np.inexact)) else str(x))
        content_length = max(len(cols[idx]), max(
            [my_len(tpl[idx]) for tpl in data]))
        widths[idx] = min(width, content_length) + 2

    # string to hold the entire table data
    table = ""
    # create header row
    for idx, col in enumerate(cols):
        arr = "<" if idx == 0 else ">"
        table += f"{col:{arr}{widths[idx]}}|"
    table += "\n"
    for idx, _ in enumerate(cols):
        table += f"{'':-<{widths[idx]}}+"
    table += "\n"

    # add every data tuple to the table, formatted according to type
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

    # print the whole thing
    print(table)


def save_pickle(obj, filename, check_file_ending=True, compress=None):
    """
    Binary dump the file into the provided location
    """
    if check_file_ending and not filename.endswith(".joblib"):
        filename += ".joblib"
    with open(filename, mode="wb") as fh:
        joblib.dump(obj, fh)


def load_pickle(filename, check_file_ending=True):
    """
    Load the binary file back from the supplied filename
    """
    if check_file_ending and not filename.endswith(".joblib"):
        filename += ".joblib"
    try:
        with open(filename, mode="rb") as fh:
            obj = joblib.load(fh)
    except IOError as e:
        obj = None
        print("Could not load file\n", e, file=sys.stderr)

    return obj
