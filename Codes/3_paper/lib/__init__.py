"""Shared helpers of the paper scripts. Importing the package puts Codes/ (for `common`) and the Didisheim replication folder (for
its `newsshock` functions) on sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "Codes") not in sys.path:
    sys.path.insert(0, str(ROOT / "Codes"))
DIDISHEIM = ROOT / "Codes/2_replication/didisheim"
if str(DIDISHEIM) not in sys.path:
    sys.path.append(str(DIDISHEIM))                    # appended: that folder's report.py must not shadow 3_paper/report.py
DIDISHEIM_RESULTS = ROOT / "Results/replication/didisheim_extended"   # the replication rerun on the extended sample (02_msrr.py --end, --results)
WORK = ROOT / "Data/clean"                             # where the paper's caches go (ridge statistics, category list); lib.run.start points a
                                                       # partial --years run to its own folder under Results/paper/_tests
