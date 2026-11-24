"""Output folders and run manifests: write_manifest records script, arguments, sample definition, time and the specification
groups the product depends on; require_same_sample raises when an input's manifest is missing, was written under another sample
definition, or under other values of one of the specification groups it declares (definitions and parameters, not the input files'
contents). Each product declares only the groups it depends on, so a change of a paper constant does not invalidate the
replication's forecasts."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

from common.universe import DESCRIPTION, UNIVERSE_PCT
from common.config import (FIRST_YEAR, MIN_MONTH_STOCKS, MIN_NAMES, MSRR_GRID, MSRR_INIT, RET_CLIP, RIDGE_GRID, TRAIN_YEARS, VAL_YEARS,
                           Y_CLIP)

SPECS = {
    "ridge": {"ridge_grid": list(RIDGE_GRID), "penalty": "c / dimension per observation", "standardize": False, "y_clip": Y_CLIP,
              "first_year": FIRST_YEAR, "train_years": TRAIN_YEARS, "val_years": VAL_YEARS, "min_names": MIN_NAMES},   # the CKX ridge and its forecasts
    "monthly": {"ridge_target_clip": list(RET_CLIP), "min_month_stocks": MIN_MONTH_STOCKS, "msrr_grid": list(MSRR_GRID), "msrr_init": MSRR_INIT},   # the Didisheim pipeline and the monthly ridge (the clip: its training target only)
}   # 3_paper/lib/constants.py adds the group "paper" (its constants) when imported


def staging_dir(path):
    """an empty staging folder next to the final one; publish() moves it into place when the run succeeded"""
    path = Path(path)
    path = path.with_name(path.name + ".staging")   # one per target: never run the same target twice at once
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def publish(staging):
    """replace the final folder with the staging folder (the previous results survive a failed run)"""
    staging = Path(staging)
    final = staging.with_name(staging.name[:-len(".staging")])
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)


def logger():
    """print with the elapsed seconds since the script started"""
    t0 = time.time()
    return lambda *x: print(*x, f"[{time.time() - t0:.0f}s]", flush=True)


def write_manifest(path, args: dict, spec=(), **extra):
    """write the manifest to `path` (a json file, or a folder which gets run.json); atomic replace. `spec`: the specification
    groups (keys of SPECS) the product depends on; their current values are stamped and checked by every reader"""
    path = Path(path)
    fn = path / "run.json" if path.is_dir() else path
    root = f"/{Path(__file__).resolve().parents[2].name}/"                 # paths inside the project are recorded relative to it
    args = {k: v.split(root, 1)[1] if isinstance(v, str) and v.startswith("/") and root in v else v for k, v in args.items()}
    info = {"script": Path(sys.argv[0]).name, "args": args, "sample": DESCRIPTION, "universe_pct": UNIVERSE_PCT,
            "spec": {g: SPECS[g] for g in spec}, "finished": time.strftime("%Y-%m-%d %H:%M"), **extra}
    tmp = fn.with_suffix(f".{os.uname().nodename}.{os.getpid()}.tmp")   # per process: two runs may write one shared manifest
    tmp.write_text(json.dumps(info, indent=1, default=str))
    tmp.replace(fn)


def require_same_sample(fn, what, needs=()):
    """raise unless the manifest at fn exists, was written under the current sample definition, declares every specification group
    in `needs` and matches the current values of every group it declares"""
    fn = Path(fn)
    fn = fn / "run.json" if fn.is_dir() else fn
    if not fn.exists():
        raise RuntimeError(f"{what}: no manifest at {fn}; rerun the script that produces it")
    m = json.loads(fn.read_text())
    if m.get("universe_pct") != UNIVERSE_PCT or m.get("sample") != DESCRIPTION:
        raise RuntimeError(f"{what}: produced under a different sample definition ({m.get('sample')}, NYSE {m.get('universe_pct')}%); rerun it")
    spec = m.get("spec")
    if not isinstance(spec, dict) or any(g not in spec for g in needs):
        raise RuntimeError(f"{what}: its manifest does not declare the specification groups {list(needs)}; rerun it")
    for g, v in spec.items():
        if g not in SPECS:
            raise RuntimeError(f"{what}: unknown specification group '{g}' (an older manifest, or 'paper' read without lib.constants); rerun it")
        if v != json.loads(json.dumps(SPECS[g], default=str)):
            raise RuntimeError(f"{what}: produced under other '{g}' parameters ({v} vs {SPECS[g]}); rerun it")
    return m
