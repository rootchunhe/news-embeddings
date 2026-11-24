"""Build and execute CKX_replication.ipynb (run from this folder):  python 03_notebook.py [--no-exec]
The cells call functions of report.py; section 0 is one summary table, the other sections follow the paper's table numbers."""
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
md = lambda s: nb.cells.append(nbformat.v4.new_markdown_cell(s))
code = lambda s: nb.cells.append(nbformat.v4.new_code_cell(s))

md("""# Chen–Kelly–Xiu, *Expected Returns and Large Language Models*: replication on RavenPack Dow Jones-edition headlines

**Data.** RavenPack RPA 1.0 (WRDS), provider `DJ`: Dow Jones Newswires plus the press-release wires it carries and MarketWatch; headlines only, 2000–2026. The paper: Reuters articles and alerts 1996–2019:H1 (the 2026 version adds US articles to 2022).

**Sample** (`Codes/common/universe.py`): the paper's rules (one tagged company, next open-to-open return available, 5-day bag-of-words de-duplication) plus a RavenPack event category and a cap floor at the NYSE 5th percentile, which brings the number of stocks with news per month close to the paper's 2,593 (the counts are in the last section). No content filter: the Dow Jones edition carries automatically generated items (insider filings, chart commentary) that Reuters does not, which is why it has about twice the paper's stories per month.

**Pipeline.** Frozen-LLM mean-pooled headline embeddings (Llama-2-13B as in the paper, RoBERTa-large as the cheap baseline) → rolling ridge (train T−8..T−3, validation T−2..T−1 with labels realized before T, test T) → daily quintile/decile long-short on open-to-open returns traded at the next open (news before 09:00 ET trades the same day). Periods: `paper` = 2004-01..2019-06 (the paper's window), `extended` = 2004-2025 (the test years of the paper analyses). Returns in the leg tables are annualized, in percent. Alphas use the strategy marked at the close so the window matches close-to-close factors; costs are 10/20 bp per 100% turnover with turnover = ½Σ|Δw|. Paper numbers come from `Codes/common/paper_numbers.py`.""")
code(f"import sys; from pathlib import Path; assert Path(sys.prefix).resolve() == Path({sys.prefix!r}).resolve(), sys.executable   # the project interpreter")
code("""from pathlib import Path
import pandas as pd
from IPython.display import display
import report
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 60); pd.set_option('display.precision', 2)
here = Path.cwd().resolve(); ROOT = next(q for q in [here, *here.parents] if (q / 'Results' / 'replication' / 'ckx').exists()); R = ROOT / 'Results/replication/ckx'
display(report.runs(R))""")
md("""## 0. Summary: paper vs ours, 2004–2019:H1""")
code("""display(report.summary(R).round(2))""")
md("""## Table 3. Cross-sectional correlation between forecasts and next-day returns, by year""")
code("""report.plot_correlation_by_year(R)
for m, lab in report.models(R).items(): print('---', lab); display(report.table(R, m, 'T03_correlation', index_col=0).head(5).T.round(2))""")
md("""## Tables 5 and 6. Quintile and decile portfolios (annualized return, volatility, Sharpe) and Figure 5, cumulative returns""")
code("""for m, lab in report.models(R).items():
    print('---', lab, 'quintiles (Table 5)'); display(report.table(R, m, 'T05_quintile', header=[0, 1], index_col=0).round(2))
    print('---', lab, 'deciles (Table 6)'); display(report.table(R, m, 'T06_decile', header=[0, 1], index_col=0).round(2))
report.plot_cumulative(R)""")
md("""## Tables 8-9. Transaction costs with EWCT turnover control (paper: the net Sharpe of each model, Table 9, articles and alerts; LLaMA3 gross and net, Table 8)""")
code("""report.plot_ewct(R)
display(report.ewct(R).round(2))""")
md("""## Table 18 and Figure 10. Delayed trading (the day-0 quintiles held with delay k) and news assimilation by size""")
code("""report.plot_assimilation(R)
for m, lab in report.models(R).items(): print('---', lab); display(report.table(R, m, 'T18_delay').set_index(['delay', 'weight']).unstack('weight').round(2))""")
md("""## Table 17 and Figure 9. Short-term reversal and the double sort""")
code("""for m, lab in report.models(R).items():
    print('---', lab, 'Table 17: quintiles on the past 5-day return'); display(report.table(R, m, 'T17_reversal', header=[0, 1, 2], index_col=0).round(2))
    print('---', lab, 'Figure 9: 5x5 sort, EW mean return bp/day'); display(report.table(R, m, 'F09_double_sort', index_col=0))""")
md("""## Tables 20–21. Alphas on the full factor set (DMRS hedged factors, q5, momentum, reversals), close-valued returns""")
code("""report.plot_loadings(R)
for m, lab in report.models(R).items():
    print('---', lab, 'all factors'); display(report.table(R, m, 'T20_21_alphas', index_col=0).T.round(3))""")
md("""## 2024 version, Tables 5–6. The sentiment (logistic) model""")
code("""for m, lab in report.models(R).items():
    acc = report.table(R, m, 'v2024_T05_accuracy', index_col=0)
    if acc is None: print('---', lab, ': sentiment model not estimated in this run'); continue
    print('---', lab, 'accuracy by year'); display(acc.round(2)); print('---', lab, 'sentiment portfolios'); display(report.table(R, m, 'v2024_T06_sentiment', header=[0, 1], index_col=0).round(2))""")
md("""## Out-of-sample extension, Table 13 (knowledge cutoff), Table 4 and Figure 7 (model comparison)""")
code("""display(report.periods(R).round(2))
for m, lab in report.models(R).items():
    t = report.table(R, m, 'T13_cutoff', 'extended', header=[0, 1], index_col=0)
    if t is not None: print('---', lab, 'Table 13: EW quintile before / after the knowledge cutoff'); display(t.round(2))
cm = report.cross_model(R)
if cm is not None: print('--- Table 4 (DM, row RoBERTa, column LLaMA2, positive = the column model better) and Figure 7 (strategy correlation), by period'); display(cm.round(3))""")
md("""## Sample counts by year""")
code("""for m, lab in report.models(R).items(): print('---', lab); display(report.table(R, m, 'counts', 'extended', index_col=0).round(0))""")

out = HERE / "CKX_replication.ipynb"
nbformat.write(nb, out)
print("written", out)
if not a.no_exec:
    from nbclient import NotebookClient
    os.environ.setdefault("JUPYTER_CONFIG_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "jupcfg_empty"))
    Path(os.environ["JUPYTER_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, out)
    print("executed", out)
