"""Miner2: fix + screen macro-sensitivity beta family (per-symbol Series rolling cov).

Also adds: downside/upside basket correlation gap (retry with robust per-symbol loop).
Gates: |IC1| >= 0.0070 and |ICIR1| >= 0.0840, min 8 names.
"""
import sys, time, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner2_fast as F

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
CP, RET, MAC = cache["close"], cache["ret"], cache["macro"]
idx = CP.index
fwd = {h: CP.shift(-h) / CP - 1.0 for h in (1, 5, 10)}
N_CELLS = len(idx) * len(SYMBOLS)
t0 = time.time()

MACR = MAC.pct_change().reindex(idx)
cands = {}

# ---- A. macro betas via per-symbol Series rolling cov ----
for col in MACR.columns:
    mr = MACR[col]
    for win in (20, 60):
        var_m = mr.rolling(win).var()
        beta = pd.DataFrame({s: RET[s].rolling(win).cov(mr) for s in SYMBOLS}) / (var_m + 1e-12)
        beta = beta.reindex(idx)
        cands[f"beta_{col.lower()}_{win}d"] = beta
print(f"macro betas built ({time.time()-t0:.1f}s)")

# ---- B. basket downside/upside corr gap (robust per-symbol loop) ----
BASK = RET.mean(axis=1)


def cond_corr(win, side):
    cols = {}
    for s in SYMBOLS:
        r = RET[s].values
        b = BASK.values
        out = np.full(len(r), np.nan)
        for i in range(win, len(r)):
            seg_b = b[i - win:i]
            m = seg_b < 0 if side == "dn" else seg_b > 0
            if m.sum() >= 5:
                a = r[i - win:i][m]
                bb = seg_b[m]
                if a.std() > 0 and bb.std() > 0:
                    out[i] = np.corrcoef(a, bb)[0, 1]
        cols[s] = out
    return pd.DataFrame(cols, index=RET.index)


for win in (60, 120):
    up = cond_corr(win, "up")
    dn = cond_corr(win, "dn")
    cands[f"up_corr_{win}d"] = up
    cands[f"dn_corr_{win}d"] = dn
    cands[f"dn_up_gap_{win}d"] = dn - up
print(f"cond corr built ({time.time()-t0:.1f}s)")

# ---- screen ----
res = []
for name, panel in cands.items():
    panel = panel.reindex(idx)
    try:
        cov = float(panel.notna().sum().sum()) / N_CELLS
        to = F.turnover10(panel)
        ic1 = F.fast_ic(panel, fwd[1])
        passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
        if passed or True:
            ic5 = F.fast_ic(panel, fwd[5])
            ic10 = F.fast_ic(panel, fwd[10])
            print(f"{name:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
                  f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} | IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
        res.append({"name": name, "cov": cov, "to": to, "ic1": ic1, "passed": passed})
    except Exception as e:
        print(f"{name}: ERROR {e}")

print(f"\nscreen done in {time.time()-t0:.1f}s | {sum(r['passed'] for r in res)} passed / {len(cands)}")
for r in res:
    if r["passed"]:
        print("PASSED:", r["name"], f"IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f}")
