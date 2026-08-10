"""Miner2 combined screen #5: all families with calendar-aware rolling windows.

Key fix: the panel cache aligns 15 assets to a UNION calendar (~50% NaN per
symbol for non-crypto). Rolling stats must use min_periods (~win/2) computed on
available observations, otherwise almost every window is NaN.
Beta family computed with explicit finite-pair rolling loop.
"""
import sys, time, json, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner2_fast as F

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
t0 = time.time()
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
CP, OP = cache["close"], cache["open"]
HP, LP, V = cache["high"], cache["low"], cache["vol"]
RET, MAC = cache["ret"], cache["macro"]
idx = CP.index
fwd = {h: CP.shift(-h) / CP - 1.0 for h in (1, 2, 3, 5, 10, 20)}
N_CELLS = len(idx) * len(SYMBOLS)
LOG = np.log(CP / CP.shift(1))
cands = {}
MP = lambda w: max(5, w // 2)  # min_periods

# ---------- A. macro betas (finite-pair rolling loop) ----------
MACR = MAC.pct_change().reindex(idx)


def rolling_beta(ret, macr, win, min_obs):
    out = np.full(len(ret), np.nan)
    r, m = ret.values, macr.values
    for i in range(win, len(r)):
        a = r[i - win:i]
        b = m[i - win:i]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_obs:
            continue
        aa, bb = a[ok], b[ok]
        if bb.std() <= 1e-12 or aa.std() <= 1e-12:
            continue
        out[i] = np.cov(aa, bb)[0, 1] / bb.var()
    return out


for col in MACR.columns:
    mr = MACR[col]
    for win in (20, 60):
        beta = pd.DataFrame({s: rolling_beta(RET[s], mr, win, MP(win)) for s in SYMBOLS}, index=idx)
        cands[f"beta_{col.lower()}_{win}d"] = beta

# ---------- B. basket conditional correlations ----------
BASK = RET.mean(axis=1)


def cond_corr(win, side):
    cols = {}
    for s in SYMBOLS:
        r = RET[s].values
        b = BASK.values
        out = np.full(len(r), np.nan)
        for i in range(win, len(r)):
            seg_b = b[i - win:i]
            m = (seg_b < 0) if side == "dn" else (seg_b > 0)
            if m.sum() < 5:
                continue
            a = r[i - win:i]
            ok = m & np.isfinite(a)
            if ok.sum() < 5:
                continue
            aa, bb = a[ok], seg_b[ok]
            if aa.std() > 0 and bb.std() > 0:
                out[i] = np.corrcoef(aa, bb)[0, 1]
        cols[s] = out
    return pd.DataFrame(cols, index=RET.index)


up60 = cond_corr(60, "up")
dn60 = cond_corr(60, "dn")
cands["up_corr_60d"] = up60
cands["dn_corr_60d"] = dn60
cands["dn_up_gap_60d"] = dn60 - up60

# ---------- C. tail shape (min_periods) ----------
for nd in (20, 60, 120):
    cands[f"skew_{nd}d"] = RET.rolling(nd, min_periods=MP(nd)).skew()
    cands[f"kurt_{nd}d"] = RET.rolling(nd, min_periods=MP(nd)).kurt()
    cands[f"nskew_{nd}d"] = -RET.rolling(nd, min_periods=MP(nd)).skew()

# ---------- D. downside vol ratio ----------
def downside_ratio(nd):
    dn = RET.clip(upper=0.0)
    dn_sd = np.sqrt((dn ** 2).rolling(nd, min_periods=MP(nd)).mean())
    return dn_sd / (RET.rolling(nd, min_periods=MP(nd)).std() + 1e-12)


cands["downside_ratio_20d"] = downside_ratio(20)
cands["downside_ratio_60d"] = downside_ratio(60)

# ---------- E. range / body ----------
rg = (HP - LP).replace(0, np.nan)
pos = (CP - LP) / rg
body = (CP - OP).abs() / rg
cands["range_pos_5d"] = pos.rolling(5, min_periods=3).mean()
cands["range_pos_20d"] = pos.rolling(20, min_periods=MP(20)).mean()
cands["body_ratio_5d"] = body.rolling(5, min_periods=3).mean()
cands["body_ratio_20d"] = body.rolling(20, min_periods=MP(20)).mean()
cands["nbody_ratio_5d"] = -body.rolling(5, min_periods=3).mean()

# ---------- F. volume ----------
VOL = V.astype(float)
vol20 = VOL.rolling(20, min_periods=MP(20)).mean()
cands["vol_z_20d"] = (VOL - vol20) / (VOL.rolling(60, min_periods=MP(60)).std() + 1e-9)
cands["nvol_z_20d"] = -cands["vol_z_20d"]
cands["vol_trend_20_60"] = vol20 / (VOL.rolling(60, min_periods=MP(60)).mean() + 1e-9)
cands["amihud_20d"] = (RET.abs() / (VOL + 1e-9)).rolling(20, min_periods=MP(20)).mean()
cands["amihud_60d"] = (RET.abs() / (VOL + 1e-9)).rolling(60, min_periods=MP(60)).mean()
cands["pv_corr_20d"] = RET.rolling(20, min_periods=MP(20)).corr(VOL.pct_change())

# ---------- G. drawdown ----------
cands["max_dd_60d"] = CP.rolling(60, min_periods=MP(60)).max() / CP - 1.0
cands["max_dd_252d"] = CP.rolling(252, min_periods=100).max() / CP - 1.0
dist252 = CP / CP.rolling(252, min_periods=100).max() - 1.0
cands["dist_252_high"] = dist252
cands["ndist_252_high"] = -dist252

# ---------- H. breadth ----------
up = (RET > 0).astype(float)
cands["up_frac_20d"] = up.rolling(20, min_periods=MP(20)).mean()
cands["up_frac_60d"] = up.rolling(60, min_periods=MP(60)).mean()

# ---------- I. vol term-structure ----------
v5 = RET.rolling(5, min_periods=3).std()
v20 = RET.rolling(20, min_periods=MP(20)).std()
v60 = RET.rolling(60, min_periods=MP(60)).std()
cands["vol_ratio_5_20"] = v5 / (v20 + 1e-12)
cands["nvol_ratio_5_20"] = -(v5 / (v20 + 1e-12))
cands["vol_ratio_20_60"] = v20 / (v60 + 1e-12)

# ---------- J. overnight / intraday ----------
gap = OP / CP.shift(1) - 1.0
intra = CP / OP - 1.0
cands["gap_1d"] = gap
cands["intra_1d"] = intra
cands["gap_5d"] = gap.rolling(5, min_periods=3).mean()
cands["intra_5d"] = intra.rolling(5, min_periods=3).mean()
cands["gap_20d"] = gap.rolling(20, min_periods=MP(20)).mean()
cands["intra_20d"] = intra.rolling(20, min_periods=MP(20)).mean()

# ---------- K. trend variants / calibration ----------
cands["mom_10d_skip5"] = CP.shift(5) / CP.shift(15) - 1.0
cands["vol_20d"] = v20 * np.sqrt(252)
cands["sharpe_60d"] = RET.rolling(60, min_periods=MP(60)).mean() * 252 / (v60 * np.sqrt(252) + 1e-12)
cands["kaufman_er_20d"] = (CP - CP.shift(20)).abs() / (LOG.abs().rolling(20, min_periods=MP(20)).sum() + 1e-12)
cands["z_close_ma20"] = (CP - CP.rolling(20, min_periods=MP(20)).mean()) / (CP.rolling(20, min_periods=MP(20)).std() + 1e-12)
upn = RET.clip(lower=0).rolling(14, min_periods=7).mean()
dnn = (-RET.clip(upper=0)).rolling(14, min_periods=7).mean()
cands["rsi_14"] = 100.0 - 100.0 / (1.0 + upn / (dnn + 1e-12))
cands["percB_20"] = (CP - CP.rolling(20, min_periods=MP(20)).min()) / (CP.rolling(20, min_periods=MP(20)).max() - CP.rolling(20, min_periods=MP(20)).min() + 1e-12)

# ---------- L. vol-scaled reversal ----------
cands["rev_vol_1d"] = -(CP / CP.shift(1) - 1.0) / (v20 + 1e-12)
cands["rev_vol_5d"] = -(CP / CP.shift(5) - 1.0) / (v20 + 1e-12)

# ---------- M. weekday seasonality ----------
weekday_eff = pd.DataFrame(index=idx, columns=SYMBOLS, dtype=float)
wd = RET.index.dayofweek
for s in SYMBOLS:
    for d in range(5):
        m = wd == d
        seg = RET[s].loc[m]
        rm = seg.rolling(20, min_periods=8).mean()
        weekday_eff.loc[m, s] = rm
cands["weekday_eff"] = weekday_eff

print(f"built {len(cands)} candidates ({time.time()-t0:.1f}s)")

res = []
for name, panel in cands.items():
    panel = panel.reindex(idx)
    try:
        cov = float(panel.notna().sum().sum()) / N_CELLS
        to = F.turnover10(panel)
        ic1 = F.fast_ic(panel, fwd[1])
        ic5 = F.fast_ic(panel, fwd[5])
        ic10 = F.fast_ic(panel, fwd[10])
        passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
        print(f"{name:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} | IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
        res.append({"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed})
    except Exception as e:
        print(f"{name}: ERROR {e}")

json.dump(res, open("scripts/miner2_screen_all_v2_results.json", "w"), indent=1, default=str)
print(f"\nscreen done in {time.time()-t0:.1f}s | {sum(r['passed'] for r in res)} passed / {len(cands)}")
for r in res:
    if r["passed"]:
        print("PASSED:", r["name"], f"IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f}")
