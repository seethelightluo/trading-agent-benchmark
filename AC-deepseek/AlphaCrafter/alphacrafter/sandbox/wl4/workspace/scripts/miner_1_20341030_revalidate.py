"""miner_1 2034-10-30 - re-validate currently effective factors on latest data window."""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, ret_panel, forward_returns,
                                 rank_ic_series, summarize_ic, decay_profile,
                                 coverage_metrics, turnover_rank, TRADABLE, MACRO)

panels = load_panels(days=6000)
closes = close_panel(panels)
rets = ret_panel(panels)
print("closes:", closes.shape, closes.index.min().date(), "..", closes.index.max().date())

# data density
valid = closes.notna()
dates_ge8 = (valid.sum(axis=1) >= 8)
print("dates with >=8 valid:", int(dates_ge8.sum()), "of", len(closes))
print("per-asset valid days:")
print(valid.sum())

# ---- effective factor 1: vol_adj_mom_accel_20x60
f1 = (closes / closes.shift(20) - 1.0 - (closes / closes.shift(60) - 1.0)) / rets.rolling(20).std()
# ---- effective factor 2: dn_mkt_beta_60d
mkt = closes[["SPX"]].mean(axis=1)
mkt_ret = mkt.pct_change()
beta = {}
for a in closes.columns:
    z = pd.concat([rets[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
    b = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    beta[a] = b
beta_df = pd.DataFrame(beta, index=closes.index)
down = rets < 0
mkt_down = mkt_ret < 0
down_beta = {}
for a in closes.columns:
    z = pd.concat([rets[a].rename("a"), mkt_ret.rename("m"), mkt_down.rename("md")], axis=1)
    zd = z[z["md"]]
    if len(zd) < 30:
        down_beta[a] = np.nan
    else:
        b = zd["a"].rolling(60).cov(zd["m"]) / zd["m"].rolling(60).var()
        down_beta[a] = b
dn_beta_df = pd.DataFrame(down_beta, index=closes.index)

def eval_factor(name, fp, expected_sign=1, label=""):
    fwd10 = forward_returns(closes, 10)
    ics_all = rank_ic_series(fp, fwd10, min_valid=8)
    print(f"\n=== {name} {label} ===")
    print("FULL period:", summarize_ic(ics_all, expected_sign))
    for tag, sl in [("R250", ics_all.index >= closes.index[-250]),
                    ("R500", ics_all.index >= closes.index[-500]),
                    ("R125", ics_all.index >= closes.index[-125])]:
        s = ics_all[sl]
        if len(s) > 20:
            print(tag, ":", summarize_ic(s, expected_sign), "n=", len(s))
    return ics_all

eval_factor("vol_adj_mom_accel_20x60", f1, 1, "REVALIDATION 2034-10-27")
eval_factor("dn_mkt_beta_60d", dn_beta_df, 1, "REVALIDATION 2034-10-27")
