"""Trader sweep v5: vectorized own-calendar factor panels, persisted ensemble weights."""
import json, os, itertools
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"
CAL = pd.DatetimeIndex([pd.Timestamp(d) for d in json.load(open("../persistent/date.json"))["trading_days"]])
REF_IDX = int(np.where(CAL == pd.Timestamp("2026-07-16"))[0][0])
END = pd.Timestamp("2026-07-15")

ENS = json.load(open("factor_ensemble.json"))
FACTOR_LIST = [f["factor_id"] for f in ENS["selected_factors"]]
DIRS = {f["factor_id"]: f["direction"] for f in ENS["selected_factors"]}
WGT = {f["factor_id"]: f["weight"] for f in ENS["selected_factors"]}
print("Ensemble:", FACTOR_LIST)
print("Weights:", WGT)

def load(sym, folder=DATA):
    df = pd.read_csv(os.path.join(folder, sym + ".csv"), parse_dates=[0])
    df.columns = [c.strip() for c in df.columns]
    dcol = [c for c in df.columns if c.lower() in ("date", "datetime")][0]
    df = df.set_index(pd.to_datetime(df[dcol])).sort_index()
    ccol = [c for c in df.columns if c.lower() == "close"][0]
    return df[ccol]

SER = {s: load(s).loc[:END] for s in ASSETS}
VIX = load("VIX", IDX).loc[:END]
CLOSE_PANEL = pd.DataFrame({s: SER[s].reindex(CAL).ffill() for s in ASSETS}).loc[:END]

# ---- vectorized factor panels (own calendar per asset) ----
FACTOR_PANELS = {}  # factor -> dict asset -> Series indexed by asset dates
def build_panels():
    rv = {f: {} for f in FACTOR_LIST}
    vix_ret = VIX.pct_change()
    vix_21 = VIX.shift(20)  # vix 20d move denominator: vix_t / vix_{t-20} - 1
    for a in ASSETS:
        s = SER[a]
        if len(s) < 130:
            continue
        r = s.pct_change()
        if "mom_120d_skip5" in FACTOR_LIST:
            rv["mom_120d_skip5"][a] = s.shift(5) / s.shift(125) - 1.0
        if "mom_10d_skip5" in FACTOR_LIST:
            rv["mom_10d_skip5"][a] = s.shift(5) / s.shift(15) - 1.0
        if "vol_of_vol20x60" in FACTOR_LIST:
            vol20 = r.rolling(20).std()
            rv["vol_of_vol20x60"][a] = vol20.rolling(60).std()
        if "vix_beta_cond_60x20" in FACTOR_LIST:
            m = pd.concat([r.rename("a"), vix_ret.rename("v")], axis=1, join="inner")
            beta = m["a"].rolling(60).cov(m["v"]) / m["v"].rolling(60).var()
            rv["vix_beta_cond_60x20"][a] = -beta * (s.reindex(m.index).shift(0) / s.reindex(m.index).shift(20) - 1.0)
    return rv

FACTOR_PANELS = build_panels()
print("panels built:", {f: len(v) for f, v in FACTOR_PANELS.items()})

def factor_cross_section(cutoff):
    """dict[factor] -> Series(asset -> last value <= cutoff on own calendar)."""
    out = {}
    for f in FACTOR_LIST:
        d = {}
        for a, ser in FACTOR_PANELS[f].items():
            v = ser[ser.index <= cutoff]
            if len(v) > 0 and np.isfinite(v.iloc[-1]):
                d[a] = float(v.iloc[-1])
        if d:
            out[f] = pd.Series(d)
    return out

def percentile_rank(vals, assets):
    out = pd.Series(0.5, index=assets)
    v = pd.Series(vals)
    if len(v) >= 8 and v.nunique() > 1:
        out.loc[v.index] = v.rank(pct=True)
    return out

def scores_at(cutoff, overlay):
    cs = factor_cross_section(cutoff)
    sc = pd.Series(0.0, index=ASSETS)
    for f in FACTOR_LIST:
        if f not in cs:
            continue
        sc = sc + WGT[f] * DIRS[f] * percentile_rank(cs[f], ASSETS)
    if overlay in ("def12", "def30") and "vix_beta_cond_60x20" in cs:
        m = CLOSE_PANEL[ASSETS].mean(axis=1)
        r20 = m.loc[cutoff] / m.shift(20).loc[cutoff] - 1
        if r20 < 0:
            k = 0.12 if overlay == "def12" else 0.30
            vb = cs["vix_beta_cond_60x20"]
            sc = sc - k * (vb - vb.mean()) / (vb.std() + 1e-12)
    return sc

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

decision_idx = [i for i in range(140, len(CAL))
                if (i - REF_IDX) % 10 == 0 and CAL[i] <= END and i + 10 <= len(CLOSE_PANEL) - 1]
print("n decisions:", len(decision_idx))

def simulate(temp, floor, cap, overlay):
    nav, prev_w = 1.0, None
    navs, dates, stats = [], [], []
    for i in decision_idx:
        sc = scores_at(CAL[i - 1], overlay)
        if sc.abs().sum() < 1e-12:
            continue
        w = weights_from_scores(sc, temp, floor, cap)
        rets10 = CLOSE_PANEL.loc[CAL[i + 10]].reindex(ASSETS) / CLOSE_PANEL.loc[CAL[i]].reindex(ASSETS) - 1.0
        r = float((w * rets10).sum())
        turnover = float((w - prev_w).abs().sum()) if prev_w is not None else float(w.abs().sum())
        cost = 0.0 if prev_w is None else 0.0003 * turnover
        nav = nav * (1 + r) - cost * nav
        navs.append(nav); dates.append(CAL[i]); stats.append((CAL[i], r, cost, turnover))
        prev_w = w
    return pd.Series(navs, index=pd.DatetimeIndex(dates)), stats

def metrics(navs, stats, since=None):
    if since is not None:
        navs = navs[navs.index >= since]
        if len(navs) < 5:
            return dict(ret=0, sharpe=0, mdd=0, to=0, n=0)
    daily = navs.pct_change().dropna()
    yrs = max((navs.index[-1] - navs.index[0]).days / 365.25, 0.5)
    ret = navs.iloc[-1] / navs.iloc[0] - 1
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    mdd = float((navs / navs.cummax() - 1).min())
    to = float(np.mean([s[3] for s in stats]))
    return dict(ret=ret, sharpe=sharpe, mdd=mdd, to=to, n=len(navs))

rows = []
for temp, floor, cap, overlay in itertools.product(
        [0.5, 0.75, 1.0], [0.0, 0.05], [0.25, None], ["none", "def12"]):
    navs, stats = simulate(temp, floor, cap, overlay)
    m = metrics(navs, stats)
    mr = metrics(navs, stats, since=pd.Timestamp("2025-07-01"))
    rows.append((temp, floor, cap, overlay, m["ret"], m["sharpe"], m["mdd"], m["to"], m["n"],
                 mr["sharpe"], mr["ret"]))

res = pd.DataFrame(rows, columns=["temp", "floor", "cap", "overlay", "ret", "sharpe", "mdd", "turnover", "n", "sharpe_r250", "ret_r250"])
res = res[res.n >= 40].sort_values("sharpe", ascending=False)
pd.set_option("display.width", 240)
print("\n=== SWEEP v5 top15 (full period, sorted by Sharpe) ===")
print(res.head(15).to_string(index=False))
print("\n=== SWEEP v5 sorted by recent-250d Sharpe top10 ===")
print(res.sort_values("sharpe_r250", ascending=False).head(10).to_string(index=False))
