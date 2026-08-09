"""Miner2 validation: short-term reversal factor family.

Idea: In a 15-name cross-asset universe, yesterday's/last-week losers tend to
bounce (short-term reversal) -- classic effect, tested on the cross-asset panel.
Factor = -(log return over last nd days) with nd in (1,2,3,5). We evaluate the
DAILY cross-sectional IC (rank) vs forward returns, ICIR, hit ratio, turnover,
coverage, horizon decay, by-year regime robustness, and VIX-regime splits.

Admission gates (15-instrument universe): |IC1| >= 0.0070, |ICIR1| >= 0.0840.
Research window: 2020-01 .. 2026-07-15 (warm-up only).
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]  # 1y warm-up for rolling windows
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

fwd = {h: F.fwd_returns(closes, h).reindex(idx) for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(idx) * len(SYMBOLS)


def run(name, panel):
    panel = panel.reindex(idx)
    cov = float(panel.notna().sum().sum()) / N_CELLS
    to = F.turnover10(panel)
    ic1 = F.fast_ic(panel, fwd[1])
    ic5 = F.fast_ic(panel, fwd[5])
    ic10 = F.fast_ic(panel, fwd[10])
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    print(f"{name:16s} cov={cov:.3f} to={to:.3f} | "
          f"IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | "
          f"IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} | IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


print("=== Reversal family ===")
cands = {}
for nd in (1, 2, 3, 5):
    cands[f"rev_{nd}d"] = -np.log(CP / CP.shift(nd))
# vol-scaled reversal (signal = negative return normalized by realized vol)
vol20 = RET.rolling(20).std()
for nd in (1, 2, 5):
    cands[f"rev_{nd}d_vs"] = -np.log(CP / CP.shift(nd)) / vol20

res = {n: run(n, p) for n, p in cands.items()}

# ---- deeper detail for rev_1d and rev_5d ----
print("\n--- decay analysis (rev_1d, rev_2d, rev_5d) ---")
for nm in ("rev_1d", "rev_2d", "rev_5d"):
    r = F.fast_ic_all(cands[nm].reindex(idx), closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    print(f"{nm}: " + ", ".join(f"h{h} IC={v['ic']:+.4f} ICIR={v['icir']:+.3f}" for h, v in r.items()))

print("\nyear      IC1     ICIR1   hit    n_dates   (rev_1d)")
panel = cands["rev_1d"].reindex(idx)
for yr in range(2021, 2027):
    lo = pd.Timestamp(f"{yr}-01-01")
    hi = pd.Timestamp(f"{yr}-12-31") if yr < 2026 else pd.Timestamp("2026-12-31")
    m = (idx >= lo) & (idx <= hi)
    r = F.fast_ic(panel.loc[idx[m]], fwd[1].loc[idx[m]])
    if r["n_dates"]:
        print(f"{yr}  {r['ic']:+.4f}  {r['icir']:+.3f}  {r['hit']:.3f}  {r['n_dates']}")

print("\n--- VIX regime split (rev_1d) ---")
vix_raw = pd.read_csv("../persistent/index_data/VIX.csv")
vix_raw["date"] = pd.to_datetime(vix_raw["date"])
vix = vix_raw.set_index("date")["close"].reindex(idx)
med = vix.median()
for label, m in [("lowVIX", vix < med), ("highVIX", vix >= med)]:
    r = F.fast_ic(panel.loc[idx[m]], fwd[1].loc[idx[m]])
    print(f"{label:8s} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f} n={r['n_dates']}")

print(f"\nreversal family done in {time.time()-t0:.1f}s | {sum(r['passed'] for r in res.values())} passed gate")