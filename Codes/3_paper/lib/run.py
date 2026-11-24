"""Scaffolding of a paper script: arguments, log, staging output folder, saved tables, manifest.
    a, log, out = start("2_ridge")                  # --model plus any --<flag> given as keyword defaults
    save(out, "decomposition", df, "title")        # Results/paper/<section>/<model>/<name>.csv (+ printed)
    finish(out, a, log, rows=len(df))              # run.json, publish, "done"
"""
import argparse

import pandas as pd

import lib
from lib import ROOT
from lib.constants import ENCODERS, PAPER, TEST_YEARS

from common.runinfo import logger, publish, staging_dir, write_manifest

RESULTS = ROOT / "Results/paper"


def start(section, model=True, **flags):
    """parse the command line: --model (unless model=False) and one --<name> per keyword (its value is the default; a bool
    default makes a store_true flag; None makes a string option); return (args, log, staging folder). `section` may be a function
    of the arguments (a results path under Results/paper, or None for a step without results: no folder). A partial --years run writes its results and its caches under
    Results/paper/_tests/<years>/, apart from the formal run"""
    p = argparse.ArgumentParser()
    if model:
        p.add_argument("--model", default="roberta-large", choices=list(ENCODERS))
    for name, default in flags.items():
        if isinstance(default, bool):
            p.add_argument(f"--{name}", action="store_true")
        else:
            p.add_argument(f"--{name}", default=default, type=str if default is None else type(default))
    a = p.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    base = RESULTS
    if partial(a):                                                         # a partial run never touches the formal results or caches
        base = RESULTS / "_tests" / a.years
        lib.WORK = base / "_work"
        lib.WORK.mkdir(parents=True, exist_ok=True)
    section = section(a) if callable(section) else section
    out = None if section is None else staging_dir(base / section / a.model if model else base / section)
    return a, logger(), out


def partial(a):
    return bool(getattr(a, "years", "")) and list(years(a.years)) != list(TEST_YEARS)


def years(arg):
    """'2008-2015' -> range(2008, 2016); '' -> TEST_YEARS"""
    if not arg:
        return TEST_YEARS
    lo, hi = (int(v) for v in str(arg).split("-"))
    return range(lo, hi + 1)


def save(out, name, df, title=None, digits=3, index=False):
    """write <name>.csv into the staging folder and print it under a title"""
    df.to_csv(out / f"{name}.csv", index=index)
    if title:
        print(f"\n==== {title} ====\n" + df.round(digits).to_string(index=index), flush=True)
    return df


def finish(out, a, log, **extra):
    """manifest (arguments plus extra facts), publish the staging folder, log done"""
    write_manifest(out, vars(a), spec=PAPER, **extra)
    publish(out)
    log("done")
