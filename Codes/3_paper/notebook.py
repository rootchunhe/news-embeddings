"""Build and execute paper_results.ipynb from Results/paper (run from Codes/3_paper/):  ../../env/bin/python notebook.py [--no-exec]
The notebook shows every table and figure of report.py in paper order; figures are also saved as Results/paper/figures/*.pdf."""
import argparse
import os
import sys
from pathlib import Path

import nbformat

p = argparse.ArgumentParser()
p.add_argument("--no-exec", action="store_true")
a = p.parse_args()
HERE = Path(__file__).resolve().parent
nb = nbformat.v4.new_notebook()


def md(s):
    nb.cells.append(nbformat.v4.new_markdown_cell(s))


def code(s):
    nb.cells.append(nbformat.v4.new_code_cell(s))


def show(expr):
    """a cell displaying a table or a (nested) dict of panels with the notebook's helper `tables`"""
    code(f"tables({expr})")


md("""# The economic information in news embeddings — results""")
code(f"import sys; from pathlib import Path; assert Path(sys.prefix).resolve() == Path({sys.prefix!r}).resolve(), sys.executable   # the project interpreter")
code("""from pathlib import Path
import pandas as pd
import report
pd.set_option('display.width', 230); pd.set_option('display.max_columns', 40)
here = Path.cwd().resolve(); ROOT = next(q for q in [here, *here.parents] if (q / 'Results' / 'paper').exists()); R = ROOT / 'Results/paper'
def tables(x, title=''):
    if isinstance(x, dict):
        for k, v in x.items():
            tables(v, f'{title} / {k}' if title else str(k))
    else:
        if title:
            print(title)
        display(x.round(3))
report.runs(R)""")

md("""## Section 3. What predicts
**Table 1.** Event-category benchmarks and the embedding forecasts""")
show("report.table1(R)")
md("""**Figure 1.** Event drift by event group""")
code("""report.plot_drift(R);""")
md("""**Table 2.** The embedding forecast beyond the category mean""")
show("{'Fama-MacBeth': report.table_fm(R), 'category mean + residual ridge': report.table_residual(R)}")
md("""**Table 3.** Recovering the earnings surprise from headlines: Panel A explains the SUE rank, Panel B predicts the next-day return""")
show("report.table_sue(R)")
md("""**Table 4.** The firm component, valuations and growth forecasts""")
show("{'split sample': report.representation(R, inputs=report.VALUATION_INPUTS), 'fitted on the previous window': report.representation(R, 'previous window', inputs=report.VALUATION_INPUTS)}")
md("""**Table 5.** Firm networks and industry classifications""")
show("report.networks(R, report.MODELS, display=True)")
md("""**Table 6.** What the ridge forecast uses (content split), with the appendix's centroid decomposition (one-way category split)""")
show("{'content split': report.table_split(R), 'one-way category split': report.table_centroid(R)}")

md("""## Section 4. Horizons
**Table 7.** The level and the deviation of each forecast across horizons h (Table A14: the full decomposition)""")
show("report.horizon_table(R, report.TIMING_PARTS, True)")
md("""**Figure 2.** Tables 7 and A14 at every horizon: (a) the daily forecast, h trading days after the trade, (b) its monthly average and (c) the monthly-trained forecast, h months after the signal month""")
code("""report.plot_horizons(R);""")

md("""## Section 5. Pricing
**Table 8.** DKPT's residualized-embedding strategy decomposed (fixed original rule)""")
show("report.table_attribution(R)")
md("""**Figure 3.** The decomposition by horizon h""")
code("""report.plot_attribution(R);""")
md("""**Figure 4.** The MSRR strategy on each input of Table A16 (weights re-optimized at every horizon). Upper row: the embedding, its level and deviation, the level and the firm identity 36 months earlier. Lower row: the embedding, the level and the firm identity 36 months earlier on JKP residualization (solid) and with industry indicators as well (dashed)""")
code("""report.plot_standalone(R);""")

md("""## Appendix""")
show("{'subsamples': report.table_subsamples(R), 'differences from the 8-year category mean': report.table_tests(R), 'transaction costs': report.table_costs(R), 'windows': report.table(R, '1_category', 'windows'), "
     "'spanning': report.table_spanning(R)}")
code("""report.plot_stability(R); report.plot_stability(R, rows="B", name="figA_stability_coarse");""")
md("""**Size groups.** The decile long-short path of the 8-year category mean within micro, small and big stocks; the monthly average of the daily forecast (Table 7, Panel B) and the strategy decomposition (Figure 3) by the size group of the signal month""")
code("""report.plot_drift_size(R); report.plot_horizons_size(R); report.plot_attribution_size(R);""")
show("report.table_drift(R)")
show("{m: report.table(R, '2_probe', 'summary', m) for m in report.models(R, '2_probe')}")
show("report.table_signals(R)")
show("{'decomposition': report.table_decomposition_full(R), 'groups': report.table_groups(R), 'factors': report.table_factors(R)}")
show("{'rewrites': report.table_rewrites(R), 'spectrum': report.spectrum_bands(R)}")
code("""report.plot_spectrum(R);""")
show("report.horizon_table(R, report.CELLS, False)")
md("""**Figures A6-A8, A11.** The level and the deviation of each forecast of Table 7 (A6-A8) and of the monthly MSRR signal (A11) as signals: rank IC and joint Fama-MacBeth coefficients after the JKP characteristics, by horizon h""")
code("""report.plot_persistence(R);""")
show("{'attribution (re-optimized)': report.table_attribution(R, 're-optimized'), 'joint FM': report.joint_fm(R).rename(index=report.JOINT_FM_TEX), 'timing': report.timing(R), 'standalone': report.standalone(R).rename(index=report.STANDALONE_NAMES)}")
md("""**Figure A12.** The deviation's coefficient in Table 9's regressions (the joint FM above) at every lag k""")
code("""report.plot_old_news(R);""")
md("""**Tables C1-C7.** Tables 2, 6, 7, 8 and A16 from the January after each model's pretraining cutoff; Tables 4 and 5 on the windows from that year""")
show("{'C1 FM': report.table_fm(R, 'fm_post'), 'C1 residual ridge': report.table_residual(R, 'residual_post'), 'C2': report.table_split(R, 'post')}")
show("{'C3': report.horizon_table(R, report.TIMING_PARTS, True, 'post')}")
show("{'C4': report.table_attribution(R, name='attribution_post'), 'C5': report.standalone(R, 'Sharpe_post').rename(index=report.STANDALONE_NAMES), "
     "'C6 split sample': report.representation(R, first=report.POST_YEAR), 'C6 previous window': report.representation(R, 'previous window', first=report.POST_YEAR)}")
show("{'C7 ' + lab: report.networks(R, {m: lab}, report.POST_YEAR[m], display=True) for m, lab in report.MODELS.items()}")
md("""**Table 10, Tables B1-B7, Figures 5, 6, A13 and A14.** The main results of every encoder (Table 10); Tables 2, 6, 7, 8, A16 (by signal years), 4 and 5 for ChronoGPT, pretrained on text to 1999; the cumulative returns of the MSRR strategy and of its level and deviation (Figure 5) and of the daily forecast portfolio (Figure A7); the rolling daily rank ICs of the level and the deviation (Figure 6) and their monthly counterpart from the MSRR signal (Figure A6)""")
from lib import ROOT
from lib.constants import ROBUST_MODELS
if (ROOT / f"Data/clean/run_pred_{next(iter(ROBUST_MODELS))}.json").exists():   # Table 10 and the B tables once the robustness encoder's forecasts exist
    code("""from lib.constants import ROBUST_MODELS, SEGMENTS
(ROBUST,) = ROBUST_MODELS
seg = lambda f: {report.segment_label(ROBUST, s): f(s) for s in SEGMENTS[ROBUST]}""")
    show("{'Table 10 ' + report.segment_label(ROBUST, 'full'): report.encoder_summary(R), 'Table 10 ' + report.segment_label(ROBUST, 'post'): report.encoder_summary(R, 'post')}")
    show("{'B1 FM': seg(lambda s: report.table_fm(R, f'fm_{s}', ROBUST_MODELS)), 'B1 residual ridge': seg(lambda s: report.table_residual(R, f'residual_{s}', ROBUST_MODELS)), 'B2': seg(lambda s: report.table_split(R, s, ROBUST_MODELS))}")
    show("{'B3': seg(lambda s: report.horizon_table(R, report.TIMING_PARTS, True, s, ROBUST_MODELS))}")
    show("{'B4': seg(lambda s: report.table_attribution(R, name=f'attribution_{s}', encoders=ROBUST_MODELS)), 'B5': seg(lambda s: report.standalone(R, f'Sharpe_{s}', ROBUST_MODELS).rename(index=report.STANDALONE_NAMES)), "
         "'B6 split sample': report.representation(R, encoders=ROBUST_MODELS), 'B6 previous window': report.representation(R, 'previous window', encoders=ROBUST_MODELS)}")
    show("{'B7 windows from ' + str(first): report.networks(R, ROBUST_MODELS, first, display=True) for first in [report.FIRST_YEAR, SEGMENTS[ROBUST]['post'][0]]}")
    md("""**Section 7.2, footnote.** Table 11's relative IC ratio over all firm-days with ChronoGPT as the control in place of RoBERTa""")
    show("{'ChronoGPT as the control': report.cutoff_control_ratio(R)}")
    code("""report.plot_cutoff_returns(R); report.plot_cutoff(R); report.plot_cutoff_monthly(R); report.plot_cutoff_returns_daily(R);""")
else:
    md("Not available in this run: the forecasts of the robustness encoder do not exist yet.")

out = HERE / "paper_results.ipynb"
nbformat.write(nb, out)
print("written", out)
if not a.no_exec:
    from nbclient import NotebookClient
    os.environ.setdefault("JUPYTER_CONFIG_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "jupcfg_empty"))   # an empty config dir: no user extensions
    Path(os.environ["JUPYTER_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    try:
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}, allow_errors=True).execute()
    finally:
        nbformat.write(nb, out)
    errors = [c for c in nb.cells if c.cell_type == "code" and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    print("executed", out, f"({len(errors)} cells with errors)" if errors else "")
    if errors:
        raise SystemExit(1)
