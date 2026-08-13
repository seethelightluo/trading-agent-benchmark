"""miner_2 re-validation of the 3 currently EFFECTIVE library factors on data
through 2033-10-14 (full window 2020-01-01..2033-10-14 + recent 2y window).
Admission gate (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from miner2_20331017_common import (load_price_panel, load_obs_panel, rets,
                                    ic_series, summarize_ic, decay_analysis,
                                    turnover_10d, coverage_stats, roll_beta)

px = load_price_panel()
r = rets(px)
mkt = r.mean(axis=1)  # equal-weight cross-asset market

# ---------- factor 1: vol_adj_mom_accel_20x60 ----------
mom20 = px / px.shift(20) - 1.0
mom60 = px / px.shift(60) - 1.0
vol20 = r.rolling(20).std(ddof=0)
f1 = (mom20 - mom60) / vol20

# ---------- factor 2: dn_mkt_beta_60d ----------
dn = np.minimum(mkt, 0.0)
f2 = roll_beta(r, dn, 60, 40)

# ---------- factor 3: rate_beta_cn10y_60d ----------
cn10y = px["CN10Y"]
cn_ret = cn10y.pct_change()
f3 = roll_beta(r, cn_ret, 60, 40)

factors = {
    "vol_adj_mom_accel_20x60": f1,
    "dn_mkt_beta_60d": f2,
    "rate_beta_cn10y_60d": f3,
}

fwd10 = px.shift(-10) / px - 1.0
recent_cut = pd.Timestamp("2031-10-14")

for name, fpanel in factors.items():
    ic = ic_series(fpanel, fwd10)
    full = summarize_ic(ic, "full")
    rec = summarize_ic(ic[ic.index >= recent_cut], "recent2y")
    dec = decay_analysis(fpanel, px)
    to = turnover_10d(fpanel)
    cov = coverage_stats(fpanel)
    print("=" * 70)
    print(f"FACTOR {name}")
    print(" FULL :", json.dumps(full))
    print(" RECENT2y:", json.dumps(rec))
    print(" decay(h1..20):", {k: round(v, 4) for k, v in dec.items()})
    print(" turnover_10d_rank:", round(to, 3), "| coverage:", json.dumps(cov))

# pairwise library correlations on common valid dates (for provenance)
print("\n-- pairwise signal correlation (full window, h=10 IC series) --")
names = list(factors.keys())
ics = {n: ic_series(factors[n], fwd10) for n in names}
base = pd.concat(ics, axis=1)
corr = base.corr()
print(corr.round(3).to_string())
