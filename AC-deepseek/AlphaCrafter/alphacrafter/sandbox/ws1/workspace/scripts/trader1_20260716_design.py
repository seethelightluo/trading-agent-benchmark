"""Trader design sweep v2: offline cross-sectional ensemble simulation (warm-up data only).

Sweeps weighting schemes / risk overlays on the 15-asset long-only fully-invested
portfolio with 3bps rebalance costs. All decisions use data visible through the
previous completed trading day (no lookahead).
"""
import json, os, itertools
import numpy as np
import pandas as pd

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"
CAL = pd.DatetimeIndex([pd.Timestamp(d) for d in json.load(open("../persistent/date.json"))["trading_days"]])
REF_IDX = int(np.where(CAL == pd.Timestamp("2026-07-16"))[0][0])

def load(sym, folder=DATA):
    df = pd.read_csv(os.path.join(folder, sym + ".csv"), parse_dates=[0])
    df.columns = [c.strip() for c in df.columns]
    dcol = [c for c in df.columns if c.lower() in ("date", "datetime")][0]
    df = df.set_index(pd.to_datetime(df[dcol])).sort_index()
    ccol = [c for c in df.columns if c.lower() == "close"][0]
    return df[ccol]

close = pd.DataFrame({s: load(s).reindex(CAL) for s in ASSETS}).loc[: "2026-07-15"]
vix = load("VIX", IDX).reindex(CAL).loc[: "2026-07-15"]
rets = close.pct_change()

f_mom10  = close.shift(5) / close.shift(15) - 1.0
f_mom120 = close.shift(5) / close.shift(125) - 1.0
f_vov    = rets.rolling(20).std().rolling(60).std()
vix_ret  = vix.pct_change()
beta60   = rets.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
vix_move = vix / vix.shift(20) - 1.0
f_vixb   = -beta60.multiply(vix_move, axis=0)

FACTORS = {
    "mom_120d_skip5":      (f_mom120, 1, 0.486158),
    "mom_10d_skip5":       (f_mom10,  1, 0.276633),
    "vix_beta_cond_60x20": (f_vixb,  -1, 0.184054),
    "vol_of_vol20x60":     (f_vov,    1, 0.053155),
}
DIRS = {"mom_120d_skip5": 1, "mom_10d_skip5": 1, "vix_beta_cond_60x20": -1, "vol_of_vol20x60": 1}
WGT = {"mom_120d_skip5": 0.486158, "mom_10d_skip5": 0.276633, "vix_beta_cond_60x20": 0.184054, "vol_of_vol20x60": 0.053155}

decision_idx = [i for i in range(140, len(CAL))
                if (i - REF_IDX) % 10 == 0 and CAL[i] <= pd.Timestamp("2026-07-15") and i + 10 <= len(close) - 1]
print(f"decisions: {len(decision_idx)}  from {CAL[decision_idx[0]].date()} to {CAL[decision_idx[-1]].date()}")

def percentile_rank(s):
    out = pd.Series(0.5, index=s.index)
    v = s.dropna()
    if len(v) >= 8 and v.nunique() > 1:
        out.loc[v.index] = v.rank(pct=True)
    return out

def scores_at(i):
    row = CAL[i - 1]
    sc = pd.Series(0.0, index=ASSETS)
    contrib = {}
    for name, (panel, d, w) in FACTORS.items():
        x = panel.loc[row].reindex(ASSETS)
        if x.notna().sum() < 8:
            continue
        c = w * d * percentile_rank(x)
        sc = sc + c
        contrib[name] = c
    return sc, contrib

def ew_trend(i):
    row = CAL[i - 1]
    m = close[ASSETS].mean(axis=1)
    return (m.loc[row] / m.shift(20).loc[row] - 1, m.loc[row] / m.shift(60).loc[row] - 1)

def weights_from_scores(sc, temp, floor, cap):
    s = sc - sc.mean()
    e = np.exp(np.clip(s / temp, -10, 10))
    w = e / e.sum()
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
    navs, dates, stats = [], [], []
    for i in decision_idx:
        sc, contrib = scores_at(i)
        if len(contrib) < 2:
            continue
        w = weights_from_scores(sc, temp, floor, cap)
        if overlay == "def12" or overlay == "def30":
            r20, _ = ew_trend(i)
            if r20 < 0:
                k = 0.12 if overlay == "def12" else 0.30
                vb = f_vixb.loc[CAL[i - 1]].reindex(ASSETS)
                if vb.notna().sum() >= 8:
                    sc2 = sc - k * (vb - vb.mean()) / (vb.std() + 1e-12)
                    w = weights_from_scores(sc2, temp, floor, cap)
        rets10 = close.loc[CAL[i + 10]].reindex(ASSETS) / close.loc[CAL[i]].reindex(ASSETS) - 1.0
        if prev_w is not None and minmv > 0 and float((w - prev_w).abs().sum()) < minmv:
            r = float((prev_w * rets10).sum())
            nav *= (1 + r)
            navs.append(nav); dates.append(CAL[i]); stats.append((CAL[i], r, 0.0, 0.0))
            continue
        r = float((w * rets10).sum())
        cost = 0.0 if prev_w is None else 0.0003 * float((w - prev_w).abs().sum())
        nav = nav * (1 + r) - cost * nav
        navs.append(nav); dates.append(CAL[i]); stats.append((CAL[i], r, cost, float((w - prev_w).abs().sum()) if prev_w is not None else float(w.abs().sum())))
        prev_w = w
    return pd.Series(navs, index=pd.DatetimeIndex(dates)), stats

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

res = pd.DataFrame(rows, columns=["temp","floor","cap","overlay","minmv","ret","cagr","sharpe","mdd","turnover","cost","n"])
res = res[res.n >= 40].sort_values("sharpe", ascending=False)
pd.set_option("display.width", 220)
print("\n=== SWEEP top25 (full 2020-2026, sorted by Sharpe) ===")
print(res.head(25).to_string(index=False))
print("\n=== SWEEP bottom5 ===")
print(res.tail(5).to_string(index=False))
