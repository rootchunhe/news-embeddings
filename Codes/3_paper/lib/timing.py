"""When a headline meets the market. The immediate reaction ends at the open of the trade day (day0). For news published during the
session of a trading day (09:30-16:00) it starts at that day's open, i.e. the day's open-to-close return compounded with the overnight
gap into day0 (the part of the day before the news is included); for 09:00-09:30 news the pre-open gap is added in front; for
off-hours, weekend and holiday news it is the overnight gap alone. The open-to-open returns of day0 onwards then continue without a gap."""
import numpy as np
import pandas as pd


from common.crsp import daily_panel, open_to_close

SESSION = (9.0, 16.0)         # 09:00 is the cleaning rule's cutoff: later news trades the next day
OPEN = 9.5


def immediate_reaction(S):
    """adds r_imm; S needs permno, pub_day, ts_ny and the raw gap0 column. A story
    published on a trading day between 09:00 and the open also carries that day's pre-open gap (last close to the open)"""
    cr, _ = daily_panel(["permno", "dlycaldt", "dlyret", "dlyopen", "dlyclose"])
    cr = cr.sort_values(["permno", "dlycaldt"])
    cr["intra_pub"] = open_to_close(cr)
    cr["gap_pub"] = ((1 + cr.dlyret) / (1 + cr.intra_pub) - 1).where(cr.groupby("permno").pos.shift(1) == cr.pos - 1)   # last close -> open, adjusted
    # (dlyret covers the whole day; NaN after a gap in the firm's CRSP rows, as gap0 in 1_data/04_clean.py)
    cr = cr.rename(columns={"dlycaldt": "pub_day"})[["permno", "pub_day", "intra_pub", "gap_pub"]]
    S = S.merge(cr, on=["permno", "pub_day"], how="left")
    S["gap0"] = pd.to_numeric(S.gap0, errors="coerce").astype(float)
    hour = S.ts_ny.dt.hour + S.ts_ny.dt.minute / 60
    on_trading_day = S.ts_ny.dt.normalize() == S.pub_day                        # a weekend or holiday story is mapped to a later pub_day
    intraday = on_trading_day & (hour >= SESSION[0]) & (hour < SESSION[1])
    pre_open = intraday & (hour < OPEN)
    r = np.where(intraday, (1 + np.where(pre_open, S.gap_pub, 0)) * (1 + S.intra_pub) * (1 + S.gap0) - 1, S.gap0)   # NaN when a leg is missing
    S["r_imm"] = r
    return S.drop(columns=["intra_pub", "gap_pub"])

