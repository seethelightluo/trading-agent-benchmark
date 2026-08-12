"""miner_1 cycle 2027-10-29: scan batch B - orthogonal macro-beta / acceleration / volume-confirm factor families.
Data visible through 2027-10-28 (current date 2027-10-29). No future leakage.
Builds panel fresh, evaluates gate metrics, decay, turnover, coverage, and max_abs_library_correlation.
"""
import numpy as np
import pandas as pd
import pickle, json, os

CUR = pd.Timestamp("2027-10-28")
SYMS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
        "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
GATE_IC, GATE_ICIR = 0.0070, 0.0840
FULL0 = pd.Timestamp("2021-01-01")

def load(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= CUR].set_index("date").sort_index()
    return df

close, opn, high, low, vol = {}, {}, {}, {}, {}
for s in SYMS:
    d = load(f"../persistent/stock_data/{s}.csv")
    close[s] = d["close"]; opn[s] = d["open"]; high[s] = d["high"]
    low[s] = d["low"]; vol[s] = d["volume"]
C = pd.DataFrame(close).sort_index(); O = pd.DataFrame(opn).sort_index()
H = pd.DataFrame(high).sort_index(); L = pd.DataFrame(low).sort_index()
V = pd.DataFrame(vol).sort_index()
idx = C.dropna(how="all").index
C, O, H, L, V = C.loc[idx], O.loc[idx], H.loc[idx], L.loc[idx], V.loc[idx]
ret = C.pct_change()
mac = {m: load(f"../persistent/index_data/{m}.csv")["close"] for m in MACRO}
M = pd.DataFrame(mac).sort_index().reindex(C.index)

print(f"panel: {C.shape[0]} dates {C.index.min().date()} -> {C.index.max().date()}, {C.shape[1]} assets")
print("last date macro non-null:", {m: str(M[m].dropna().index.max().date()) for m in MACRO})

def daily_ic_series(f, h):
    fwd = C.shift(-h) / C - 1.0
    out = {}
    for dt in f.index:
        ff, rr = f.loc[dt], fwd.loc[dt]
        m = ff.notna() & rr.notna()
        if m.sum() < 8:
            continue
        ic = ff[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            out[dt] = ic
    return pd.Series(out)

def evaluate(f, horizons=(1, 2, 3, 5, 10)):
    rows = []
    for h in horizons:
        s = daily_ic_series(f, h)
        if len(s) < 30:
            continue
        s = s[s.index >= FULL0]
        ic = float(s.mean()); sd = float(s.std(ddof=1))
        icir = ic / sd if sd > 0 else 0.0
        hit = float((s > 0).mean()) if ic > 0 else float((s < 0).mean())
        cut = s.index.max() - pd.Timedelta(days=365)
        s12 = s[s.index >= cut]
        ic12 = float(s12.mean())
        icir12 = float(ic12 / s12.std(ddof=1)) if len(s12) > 5 and s12.std(ddof=1) > 0 else 0.0
        rows.append(dict(h=h, ic=ic, icir=icir, hit=hit, n=len(s), ic12=ic12, icir12=icir12))
    return rows

def turnover(f, rebal=10):
    ranks = f.rank(axis=1)
    chg = []
    for i in range(rebal, len(ranks)):
        prev = ranks.iloc[i - rebal].dropna(); cur = ranks.iloc[i].dropna()
        common = prev.index.intersection(cur.index)
        if len(common) < 2:
            continue
        chg.append((cur[common] - prev[common]).abs().mean() / (len(common) - 1))
    return float(np.mean(chg)) if chg else np.nan

def coverage(f):
    tot = C.notna().sum().sum()
    val = f.notna().sum().sum()
    return val / tot if tot else np.nan

# ---- library signals for correlation (sign-agnostic; abs rho used) ----
lib = {}
lib["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
for nd in (1, 2, 3, 5):
    lib[f"rev_{nd}d"] = C.shift(nd) / C - 1.0
for nd in (1, 2, 3, 5):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    lib[f"nclv_{nd}d"] = (C - lo) / (hi - lo).replace(0, np.nan) - 0.5
lib["id_rev_1d"] = C / O - 1.0
rng = (H - L).replace(0, np.nan)
lib["nbody_1d"] = (C - L) / rng - 0.5
lib["rev_1d_vs"] = (C.shift(1) / C - 1.0) / ret.rolling(10).std().replace(0, np.nan)
vol20 = ret.rolling(20).std()
lib["vol_of_vol20x60"] = vol20.rolling(60).std()
vix_ret = M["VIX"].pct_change()
cov60 = ret.rolling(60).cov(vix_ret); var60 = vix_ret.rolling(60).var().replace(0, np.nan)
cond = (M["VIX"] > M["VIX"].rolling(20).mean()).astype(float)
lib["vix_beta_cond_60x20"] = (cov60 / var60) * cond

def max_lib_rho(f):
    best = 0.0; bestname = None
    fv = f.stack().dropna()
    for name, g in lib.items():
        gv = g.stack().dropna()
        common = fv.index.intersection(gv.index)
        if len(common) < 200:
            continue
        r = np.corrcoef(fv.loc[common], gv.loc[common])[0, 1]
        if np.isfinite(r) and abs(r) > best:
            best, bestname = abs(r), name
    return best, bestname

# ---- candidate construction ----
cand = {}
dxy_ret = M["DXY"].pct_change()
usdjpy_ret = M["USDJPY"].pct_change()
eurusd_ret = M["EURUSD"].pct_change()
usdcny_ret = M["USDCNY"].pct_change()
u10_ret = C["US10Y"].pct_change()

def beta_to(asset_ret, mkt_ret, w):
    cov = asset_ret.rolling(w).cov(mkt_ret)
    var = mkt_ret.rolling(w).var().replace(0, np.nan)
    return cov / var

# B1 macro betas
cand["dxy_beta_20d"] = beta_to(ret, dxy_ret, 20)
cand["dxy_beta_60d"] = beta_to(ret, dxy_ret, 60)
# B2 FX betas
cand["usdjpy_beta_60d"] = beta_to(ret, usdjpy_ret, 60)
cand["eurusd_beta_60d"] = beta_to(ret, eurusd_ret, 60)
cand["usdcny_beta_60d"] = beta_to(ret, usdcny_ret, 60)
# B3 rate-sensitivity
cand["us10y_beta_20d"] = beta_to(ret, u10_ret, 20)
cand["us10y_beta_60d"] = beta_to(ret, u10_ret, 60)
# B4 cross-asset panel beta (self-excluded)
pan = (C * 0)
for s in SYMS:
    others = [x for x in SYMS if x != s]
    pan[s] = C[others].pct_change().mean(axis=1)
cand["panel_beta_60d"] = beta_to(ret, pan.mean(axis=1), 60)
# B5 acceleration
mom5 = C.shift(1) / C.shift(6) - 1.0
mom20 = C.shift(5) / C.shift(25) - 1.0
mom60 = C.shift(5) / C.shift(65) - 1.0
mom10 = C.shift(2) / C.shift(12) - 1.0
cand["accel_5_20"] = mom20 - mom5
cand["accel_10_60"] = mom60 - mom10
# B6 volume confirmation of trend
volv60 = V.rolling(60).mean()
cand["vol_price_conf_20"] = np.sign(mom20) * (V.rolling(20).mean() / volv60.replace(0, np.nan) - 1.0)
cand["vol_price_conf_10"] = np.sign(mom10) * (V.rolling(10).mean() / volv60.replace(0, np.nan) - 1.0)
# B7 vix change beta (plain, no condition)
cand["vix_chg_beta_60d"] = beta_to(ret, vix_ret, 60)
# B8 macro-conditioned reversal: rev_2d * sign(dxy 20d mom)
dxy_mom20 = np.sign(M["DXY"].pct_change(20))
cand["rev_2d_x_dxy_mom"] = -(C.shift(2) / C - 1.0) * dxy_mom20
# B9 risk-reward (20d sharpe)
cand["sharpe_20d"] = ret.rolling(20).mean() / vol20.replace(0, np.nan)
# B10 vol percentile within 252d
def pctile_rank(x, w=252):
    return x.rolling(w).apply(lambda a: (a <= a[-1]).mean() if len(a) == w else np.nan, raw=True)
v60 = ret.rolling(60).std()
cand["vol_pctile_60x252"] = pctile_rank(v60)

res = {}
print("\n=== SCAN BATCH B (2021+ full sample, h=1 primary; gate |IC|>=0.007 |ICIR|>=0.084) ===")
print(f"{'name':24s} {'h':>2s} {'IC':>9s} {'ICIR':>8s} {'hit':>6s} {'n':>5s} {'IC12m':>9s} {'ICIR12m':>8s} {'cov':>6s} {'to10':>6s} {'maxrho':>7s} {'gate':>5s}")
for name, f in cand.items():
    ev = evaluate(f)
    if not ev:
        print(f"{name:24s} no data")
        continue
    best = max(ev, key=lambda r: abs(r["icir"]))
    cov = coverage(f); to = turnover(f)
    rho, rnoname = max_lib_rho(f)
    ok = "PASS" if abs(best["ic"]) >= GATE_IC and abs(best["icir"]) >= GATE_ICIR else "fail"
    print(f"{name:24s} {best['h']:>2d} {best['ic']:+9.5f} {best['icir']:+8.5f} {best['hit']:6.3f} {best['n']:5d} "
          f"{best['ic12']:+9.5f} {best['icir12']:+8.5f} {cov:6.3f} {to:6.3f} {rho:7.3f} {ok:>5s}  ({rnoname})")
    res[name] = dict(best=best, all_h=ev, cov=cov, turnover=to, max_rho=rho, rho_name=rnoname)

with open("scripts/miner1_20271029_scanB_results.json", "w") as fh:
    json.dump(res, fh, indent=1, default=float)
print("\nsaved scripts/miner1_20271029_scanB_results.json")
