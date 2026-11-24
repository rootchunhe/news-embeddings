"""Build and execute Didisheim_replication.ipynb (run from this folder):  python 03_notebook.py [--no-exec]"""
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

md("""# Didisheim–Kelly–Pourmohammadi–Tian, *The Inefficient Pricing of News*: replication on Dow Jones-edition headlines

**Paper.** Reuters articles 1996–2022 (6.7M after the single-stock, length and near-duplicate filters), embedded with E5-Mistral-7B (4,096-d), averaged to stock-months and z-scored with expanding moments; residualized each month on 132 rank-standardized JKP characteristics (and, in Section 6.5, GICS industry groups); the news-managed factors $F_{t+1}=X_t'R_{t+1}$ combined by maximum-Sharpe-ratio regression (ridge, leave-one-out λ, expanding window from 12 months). Headline results (`Codes/common/paper_numbers.py`): Figure 4 Sharpe ratios 1.1 (raw), 1.7 (demeaned), rising through the characteristics of CAPM, FF3 and FF6 to 3.1 (residual on the JKP characteristics), 3.3 with industry groups; BERT-110M on the same data 1.5; Dow Jones Newswires articles with Mistral 3.7; Table 7 (MSE approach, EW H-L) 2.68; Figure 7 small stocks 2.7, large 1.4; Figure 10: the news-shock factor keeps earning for about 18 months (Panel A: the delayed shocks held with the undelayed strategy's weights; Panel B contrasts weights re-optimized for the delay).

**Ours.** The same procedure on the sample of `Codes/common/universe.py` (single-company, de-duplicated stories, cap floor at the NYSE 5th percentile), headline embeddings from RoBERTa-large (1,024-d) and Llama-2-13B (5,120-d), signal months 2000–2022; the news shock used downstream is the residual on the characteristics (the paper's main specification). Differences: headlines instead of articles, encoders, sample period, the cap floor, GCV instead of leave-one-out for the Table 7 ridge. The control runs the identical MSRR code on the characteristics themselves.""")
code(f"import sys; from pathlib import Path; assert Path(sys.prefix).resolve() == Path({sys.prefix!r}).resolve(), sys.executable   # the project interpreter")
code("""from pathlib import Path
import pandas as pd
from IPython.display import display
import report
pd.set_option('display.width', 220); pd.set_option('display.precision', 2)
here = Path.cwd().resolve(); ROOT = next(q for q in [here, *here.parents] if (q / 'Results' / 'replication' / 'didisheim').exists()); R = ROOT / 'Results/replication/didisheim'
display(report.runs(R))""")
md("""## 0. Summary: paper vs ours""")
code("""display(report.summary(R).round(2))""")
md("""## Figure 4. MSRR news portfolios under each residualization: Sharpe ratios, CAPM alphas at 10% volatility, cumulative returns""")
code("""report.plot_fig4(R)
for m, lab in report.models(R).items(): print('---', lab); display(report.table(R, m, 'F04_msrr').round(2))""")
md("""## Table 7. Quintile sorts on the MSE forecast (Panel A) and on the MSRR weights (Panel B), equal- and capped value-weighted""")
code("""for m, lab in report.models(R).items(): print('---', lab); display(report.table(R, m, 'T07_sorts').set_index(['panel', 'weighting', 'portfolio']).round(3))""")
md("""## Figure 7. Size groups""")
code("""report.plot_fig7(R)
for m, lab in report.models(R).items(): print('---', lab); display(report.table(R, m, 'F07_size').round(2))""")
md("""## Figure 10. Trading delay; the delayed returns against a same-month seasonal factor""")
code("""report.plot_fig10(R)
for m, lab in report.models(R).items():
    print('---', lab); display(report.table(R, m, 'F10_lags').set_index('tau').T.round(2))
    print('---', lab, 'seasonality (Heston-Sadka factor)'); display(report.table(R, m, 'seasonality').set_index(['weights', 'tau']).round(2))""")
out = HERE / "Didisheim_replication.ipynb"
nbformat.write(nb, out)
print("written", out)
if not a.no_exec:
    from nbclient import NotebookClient
    os.environ.setdefault("JUPYTER_CONFIG_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "jupcfg_empty"))
    Path(os.environ["JUPYTER_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, out)
    print("executed", out)
