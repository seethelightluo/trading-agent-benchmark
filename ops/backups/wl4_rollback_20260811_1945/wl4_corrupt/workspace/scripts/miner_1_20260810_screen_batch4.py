"""miner_1 batch-4 screening (fast vectorized rank-IC) - 2026-08-10.

Screens macro-beta / conditional-beta candidates at h=10 with a fast
per-row rank correlation (Spearman) implementation. Only candidates passing
the shared gates proceed to full library eval (decay/turnover/lib-corr) and
possible persistence.
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 library_signals, max_library_corr, TRADABLE)

t0 = time.time()
panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
lib = library_signals(panels, closes, rets)
H = 10
fwd10 = forward_returns(closes, H)
print(f"data ready {time.time()-t0:.1f}s, closes {closes.shape}")


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = (z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var())
        beta[a] = b.where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


def fast_rank_ic(panel, fwd, min_valid=8):
    f_rank = panel.rank(axis=1)
    r_rank = fwd.rank(axis=1)
    dates, ics = [], []
    for dt in panel.index:
        if dt not in fwd.index:
            continue
        fr, rr = f_rank.loc[dt], r_rank.loc[dt]
        mask = fr.notna() & rr.notna()
        n = int(mask.sum())
        if n < min_valid:
            continue
        fv, rv = fr[mask].to_numpy(float), rr[mask].to_numpy(float)
        if fv.std() < 1e-12 or rv.std() < 1e-12:
            continue
        ic = float(np.corrcoef(fv, rv)[0, 1])
        if not np.isnan(ic):
            dates.append(dt)
            ics.append(ic)
    s = pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")
    return s


def pct(s):
    return s.astype(float).pct_change()


cands = {}
# plain 60d macro betas
for drv, dname in [("DXY", "dxy"), ("USDJPY", "usdjpy"), ("COPPER", "copper"),
                   ("WTI", "wti"), ("XAU", "xau"), ("BTC", "btc")]:
    if drv in panels:
        cands[f"beta_{dname}_60d"] = rolling_beta(rets, pct(panels[drv]["close"]), 60)
# conditional betas: beta * 20d driver momentum
for drv, dname in [("COPPER", "copper"), ("WTI", "wti"), ("XAU", "xau"),
                   ("BTC", "btc"), ("DXY", "dxy")]:
    if drv in panels:
        dc = panels[drv]["close"].astype(float)
        beta = rolling_beta(rets, pct(dc), 60)
        mom20 = dc / dc.shift(20) - 1.0
        cands[f"cbeta_{dname}_60x20"] = beta.mul(mom20, axis=0)
# yield-curve spread beta
cn10 = panels["CN10Y"]["close"].astype(float)
us10 = panels["US10Y"]["close"].astype(float)
cands["beta_yspread_60d"] = rolling_beta(rets, (cn10 - us10).pct_change(), 60)
# gold/stock ratio beta
ratio = panels["XAU"]["close"].astype(float) / panels["SPX"]["close"].astype(float)
cands["beta_goldspx_60d"] = rolling_beta(rets, ratio.pct_change(), 60)
# relative momentum 40d
mom40 = closes.shift(5) / closes.shift(45) - 1.0
cands["rel_mom_40d"] = mom40.sub(mom40.mean(axis=1), axis=0)
# negative-variance regime vol factor: down-day vol / up-day vol (20d), plain
r = rets
up_vol = r.where(r > 0).rolling(20).std()
dn_vol = r.where(r < 0).rolling(20).std()
cands["down_up_vol20"] = dn_vol / up_vol

print(f"built {len(cands)} candidates {time.time()-t0:.1f}s")

rows = []
for name, panel in cands.items():
    t1 = time.time()
    ic = fast_rank_ic(panel, fwd10, 8)
    icv = ic.mean()
    icir = icv / ic.std(ddof=1) if ic.std(ddof=1) > 0 else 0.0
    hit = float((np.sign(ic) == 1).mean())
    corr, key = max_library_corr(panel, lib)
    status = "PASS" if abs(icv) >= 0.007 and abs(icir) >= 0.084 else ""
    rows.append((name, icv, icir, len(ic), corr, key, status))
    print(f"{name:22s} IC={icv:+.4f} ICIR={icir:+.4f} n={len(ic):4d} "
          f"rho={corr:.3f}({key}) {status} [{time.time()-t1:.1f}s]")

print("\n--- pairwise stacked corr among candidates (|r|>0.5 flagged) ---")
from itertools import combinations
names = [r[0] for r in rows]
for a, b in combinations(names, 2):
    both = pd.concat([cands[a].stack().rename("a"), cands[b].stack().rename("b")], axis=1).dropna()
    if len(both) < 30:
        continue
    r = float(both["a"].corr(both["b"]))
    if abs(r) > 0.5:
        print(f"  HIGH {a:22s} vs {b:22s}: rho={r:+.3f} n={len(both)}")

print("\n--- passers ---")
for name, icv, icir, n, corr, key, st in rows:
    if st:
        print(f"  PASS {name:22s} IC={icv:+.4f} ICIR={icir:+.4f} n={n} rho={corr:.3f}({key})")
print(f"total {time.time()-t0:.1f}s")
