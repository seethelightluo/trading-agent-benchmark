"""Trader hyperparameter sweep v2 (vectorized): 4-factor Screener ensemble.

Precomputes factor panels once, then sweeps (temp, floor, cap, overlay, minmv)
over the 2020-2026 warm-up period. Uses the exact ensemble weights/directions
persisted in factor_ensemble.json. Long-only, 15 assets, 10-trading-day cadence.
"""
import json, os, itertools
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU",
          "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"
CAL = pd.DatetimeIndex([pd.Timestamp(d) for d in json.load(open("../persistent/date.json"))["trading_days"]])
REF_IDX = int(np.where(CAL == pd.Timestamp("2026-07-16"))[0][0])
END = pd.Timestamp("2026-07-15")

def load(sym, folder=DATA):
    df = pd.read_csv(os.path.join(folder, sym + ".csv"), parse_dates=[0])
    df.columns = [c.strip() for c in df.columns]
    dcol = [c for c in df.columns if c.lower() in ("date", "datetime")][0]
    df = df.set_index(pd.to_datetime(df[dcol])).sort_index()
    ccol = [c for c in df.columns if c.lower() == "close"][0]
    return df[ccol]

SER = {s: load(s).loc[:END] for s in ASSETS}
VIX = load("VIX", IDX).loc[:END]
P = pd.DataFrame({s: SER[s].reindex(CAL).ffill() for s in ASSETS}).loc[:END]
R = P.pct_change()
V = VIX.reindex(CAL).ffill().loc[:END]
RV = V.pct_change()

# --- factor panels (row t = value computed on data visible through day t) ---
mom120 = P.shift(5) / P.shift(125) - 1.0            # 120d momentum skipping last 5d
mom10 = P.shift(5) / P.shift(15) - 1.0              # 10d momentum skipping last 5d
vol20 = R.rolling(20).std()
volvol = vol20.rolling(60).std()                    # vol-of-vol 20x60
cov_ar = R.rolling(60).cov(RV)
var_v = RV.rolling(60).var()
beta60 = cov_ar.div(var_v, axis=0)
vix_move = V.pct_change(20)
vixbeta = -beta60 * vix_move                        # conditional vix beta (direction -1)

FACTOR_LIST = ["mom_120d_skip5", "mom_10d_skip5", "vix_beta_cond_60x20", "vol_of_vol20x60"]
PANELS = {"mom_120d_skip5": mom120, "mom_10d_skip5": mom10,
          "vix_beta_cond_60x20": vixbeta, "vol_of_vol20x60": volvol}
WGT = {"mom_120d_skip5": 0.486158, "mom_10d_skip5": 0.276633,
       "vix_beta_cond_60x20": 0.184054, "vol_of_vol20x60": 0.053155}
DIRS = {"mom_120d_skip5": 1, "mom_10d_skip5": 1, "vix_beta_cond_60x20": -1, "vol_of_vol20x60": 1}
DEF = {"XAU", "US10Y", "CN10Y"}

decision_idx = [i for i in range(140, len(CAL))
                if (i - REF_IDX) % 10 == 0 and CAL[i] <= END and i + 10 <= len(P) - 1]

# --- precompute cross-sectional rank panels per factor ---
def rank_panel(fp):
    out = pd.DataFrame(0.5, index=fp.index, columns=ASSETS)
    ranked = fp.rank(axis=1, pct=True)
    out.loc[ranked.index, ASSETS] = ranked
    return out

RP = {f: rank_panel(PANELS[f]) for f in FACTOR_LIST}

# market 20d return at each row (equal-weight)
M = P[ASSETS].mean(axis=1)
mkt20 = M / M.shift(20) - 1.0

# forward 10-day block returns from decision i: P[i+10]/P[i]-1
fwd = P.shift(-10) / P - 1.0

def scores_at(i, overlay):
    r = i - 1  # cutoff row
    sc = pd.Series(0.0, index=ASSETS)
    for f in FACTOR_LIST:
        sc = sc + WGT[f] * DIRS[f] * RP[f].loc[r]
    if overlay != "none" and mkt20.loc[CAL[r]] < 0:
        k = 0.12 if overlay == "def12" else 0.30
        vb = PANELS["vix_beta_cond_60x20"].loc[r]
        if vb.std() > 1e-12:
            sc = sc - k * (vb - vb.mean()) / vb.std()
    return sc

def weights_from_scores(sc, temp, floor, cap):
    s = sc - sc.mean()
    e = np.exp(np.clip(s / temp, -10, 10))
    w = pd.Series(e / e.sum(), index=sc.index)
    if floor > 0:
        w = (1 - floor) * w + floor / len(ASSETS)
    if cap is not None:
        for _ in range(80):
            over = float((w[w > cap] - cap).sum())
            if over <= 1e-12:
                break
            w[w > cap] = cap
            room = w[w < cap - 1e-12]
            if len(room) == 0:
                break
            w.loc[room.index] += over * room / room.sum()
    return w / w.sum()

def simulate(temp, floor, cap, overlay, minmv):
    nav, prev_w = 1.0, None
    navs, stats = [], []
    for i in decision_idx:
        sc = scores_at(i, overlay)
        w = weights_from_scores(sc, temp, floor, cap)
        r = float((w * fwd.loc[CAL[i]]).sum())
        if prev_w is not None and minmv > 0 and float((w - prev_w).abs().sum()) < minmv:
            nav *= (1 + r)
            navs.append(nav); stats.append((CAL[i], r, 0.0, 0.0))
            continue
        turnover = float((w - prev_w).abs().sum()) if prev_w is not None else 1.0
        cost = 0.0 if prev_w is None else 0.0003 * turnover
        nav = nav * (1 + r) - cost * nav
        navs.append(nav); stats.append((CAL[i], r, cost, turnover))
        prev_w = w
    return pd.Series(navs, index=pd.DatetimeIndex([s[0] for s in stats])), stats

def metrics(navs, stats):
    if len(navs) < 5:
        return dict(ret=0, cagr=0, sharpe=0, mdd=0, to=0, cost=0, n=0)
    daily = navs.pct_change().dropna()
    yrs = max((navs.index[-1] - navs.index[0]).days / 365.25, 0.5)
    ret = navs.iloc[-1] / navs.iloc[0] - 1
    cagr = (navs.iloc[-1] / navs.iloc[0]) ** (1 / yrs) - 1
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    mdd = float((navs / navs.cummax() - 1).min())
    to = float(np.mean([s[3] for s in stats]))
    cost = float(sum(s[2] for s in stats))
    return dict(ret=ret, cagr=cagr, sharpe=sharpe, mdd=mdd, to=to, cost=cost, n=len(navs))

rows = []
for temp, floor, cap, overlay, minmv in itertools.product(
        [0.5, 0.75, 1.0, 1.5], [0.0, 0.05, 0.10], [0.25, 0.35, None],
        ["none", "def12", "def30"], [0.0, 0.03]):
    navs, stats = simulate(temp, floor, cap, overlay, minmv)
    m = metrics(navs, stats)
    rows.append((temp, floor, cap, overlay, minmv, m["ret"], m["cagr"], m["sharpe"], m["mdd"], m["to"], m["cost"], m["n"]))

res = pd.DataFrame(rows, columns=["temp", "floor", "cap", "overlay", "minmv", "ret", "cagr", "sharpe", "mdd", "turnover", "cost", "n"])
res = res[res.n >= 40].sort_values("sharpe", ascending=False)
pd.set_option("display.width", 220)
print("\n=== SWEEP top25 (2020-2026, sorted by Sharpe) ===")
print(res.head(25).to_string(index=False))
print("\n=== SWEEP bottom5 ===")
print(res.tail(5).to_string(index=False))
