"""miner_2 cycle-38 screening: risk/quality/conditional-macro factor family (2026-08-13).

Candidates (computed with data visible at date t, no lookahead):
  sharpe_20           : mean/std of daily returns over 20d (risk-adjusted momentum quality)
  updown_vol_ratio_60 : std(up-day rets)/std(down-day rets) over 60d (vol asymmetry)
  overnight_mom_20    : mean(open/prev_close - 1) over 20d (overnight gap drift)
  overnight_var_share_20: var(overnight ret)/var(total ret) over 20d (overnight risk concentration)
  drawup_20           : max cumulative up-run magnitude over 20d (best-run size)
  tail_ratio_20       : (P90-P50)/(P50-P10) of daily returns over 20d (return tail asymmetry)
  dxy_up_ret_60       : mean ret on DXY-up days - mean ret on DXY-down days over 60d (dollar sensitivity)
  comm_beta_60        : beta of daily returns to commodity basket (WTI, COPPER) over 60d

Metrics: daily cross-sectional Spearman IC vs h=10 fwd return; ICIR=mean/std; hit;
coverage; 10d rank turnover; decay by horizon; max abs library correlation.
Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
print(f"grid rows: {len(GRID)}  {GRID[0]}..{GRID[-1]}")

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
HORIZON = 10
MIN_ASSETS = 8

def load_asset(sym):
    df = get_stock_daily_data(sym, days=2100)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

DATA = {s: load_asset(s) for s in ASSETS}
for s, df in DATA.items():
    print(f"  {s:10s} rows={0 if df is None else len(df)} last={0 if df is None else df.index[-1]}")

def safe_div(a, b):
    return a / np.where(np.abs(b) < 1e-12, np.nan, b)

# DXY daily (observation-only macro) for conditional factors
dxy = pd.read_csv("../persistent/index_data/DXY.csv", parse_dates=["date"])
dxy["date"] = pd.to_datetime(dxy["date"]).dt.strftime("%Y-%m-%d")
dxy = dxy.set_index("date")["close"].astype(float)
dxy_ret = dxy.pct_change()
print(f"DXY rows={len(dxy)} last={dxy.index[-1]}")

def compute_factors(df):
    if df is None or len(df) < 90:
        return None
    close = df["close"]
    ret = close.pct_change()
    out = pd.DataFrame(index=df.index)

    # 1. sharpe_20
    mu = ret.rolling(20, min_periods=10).mean()
    sd = ret.rolling(20, min_periods=10).std()
    out["sharpe_20"] = safe_div(mu, sd)

    # 2. updown_vol_ratio_60
    up = ret.where(ret > 0)
    dn = ret.where(ret < 0)
    ups = up.rolling(60, min_periods=20).std()
    dns = dn.rolling(60, min_periods=20).std()
    out["updown_vol_ratio_60"] = safe_div(ups, dns)

    # 3. overnight_mom_20
    gap = safe_div(df["open"], close.shift(1)) - 1.0
    out["overnight_mom_20"] = gap.rolling(20, min_periods=10).mean()

    # 4. overnight_var_share_20
    intra = safe_div(close, df["open"]) - 1.0
    v_o = gap.rolling(20, min_periods=10).var()
    v_i = intra.rolling(20, min_periods=10).var()
    out["overnight_var_share_20"] = safe_div(v_o, v_o + v_i)

    # 5. drawup_20: max cumulative gain from running 20d min (magnitude of best run)
    def drawup_window(i, w=20):
        lo = max(0, i - w + 1)
        seg = close.iloc[lo:i + 1].values
        if len(seg) < 10:
            return np.nan
        base = seg[0]
        return float(np.max(seg / base - 1.0))
    out["drawup_20"] = [drawup_window(i, 20) for i in range(len(close))]

    # 6. tail_ratio_20: (P90-P50)/(P50-P10) of 20d returns
    def tail_ratio(i, w=20):
        lo = max(0, i - w + 1)
        seg = ret.iloc[lo:i + 1].values
        seg = seg[~np.isnan(seg)]
        if len(seg) < 10:
            return np.nan
        p90, p50, p10 = np.percentile(seg, [90, 50, 10])
        d = p50 - p10
        if abs(d) < 1e-12:
            return np.nan
        return float((p90 - p50) / d)
    out["tail_ratio_20"] = [tail_ratio(i, 20) for i in range(len(ret))]

    # 7. dxy_up_ret_60: mean ret on DXY-up days minus mean ret on DXY-down days
    dx = dxy_ret.reindex(df.index)
    up_mask = (dx > 0)
    dn_mask = (dx < 0)
    def dxy_up_ret(i, w=60):
        lo = max(0, i - w + 1)
        seg = ret.iloc[lo:i + 1]
        um = up_mask.iloc[lo:i + 1]
        dm = dn_mask.iloc[lo:i + 1]
        ru = seg[um].mean()
        rd = seg[dm].mean()
        if pd.isna(ru) or pd.isna(rd):
            return np.nan
        return float(ru - rd)
    out["dxy_up_ret_60"] = [dxy_up_ret(i, 60) for i in range(len(ret))]

    # 8. comm_beta_60: beta to commodity basket (mean of WTI, COPPER daily returns)
    wti = DATA.get("WTI"); cop = DATA.get("COPPER")
    if wti is not None and cop is not None:
        bw = wti["close"].pct_change().reindex(df.index)
        bc = cop["close"].pct_change().reindex(df.index)
        basket = pd.concat([bw, bc], axis=1).mean(axis=1)
        def comm_beta(i, w=60):
            lo = max(0, i - w + 1)
            x = basket.iloc[lo:i + 1].values
            y = ret.iloc[lo:i + 1].values
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() < 20:
                return np.nan
            xx = x[m]; yy = y[m]
            vx = xx.var()
            if vx < 1e-12:
                return np.nan
            return float(np.cov(xx, yy)[0, 1] / vx)
        out["comm_beta_60"] = [comm_beta(i, 60) for i in range(len(ret))]

    return out

FACTORS = {}
for s in ASSETS:
    f = compute_factors(DATA[s])
    if f is not None:
        FACTORS[s] = f
print("factor frames computed for", len(FACTORS), "assets")

names = ["sharpe_20", "updown_vol_ratio_60", "overnight_mom_20", "overnight_var_share_20",
         "drawup_20", "tail_ratio_20", "dxy_up_ret_60", "comm_beta_60"]

def spearman(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    if len(xr) < 2:
        return np.nan
    sx, sy = xr.std(), yr.std()
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])

def ic_series(fname):
    ics = {}
    for t in GRID:
        xs, ys = [], []
        for s in ASSETS:
            if s not in FACTORS or s not in DATA:
                continue
            if t not in FACTORS[s].index or t not in DATA[s].index:
                continue
            idx = DATA[s].index.get_loc(t)
            j = idx + HORIZON
            if j >= len(DATA[s]):
                continue
            x = FACTORS[s].loc[t, fname]
            y = DATA[s]["close"].iloc[j] / DATA[s]["close"].iloc[idx] - 1.0
            if pd.notna(x) and pd.notna(y):
                xs.append(x); ys.append(y)
        if len(xs) >= MIN_ASSETS:
            ics[t] = spearman(xs, ys)
    return pd.Series(ics)

def ic_at_horizon(fname, h):
    ics = {}
    for t in GRID:
        xs, ys = [], []
        for s in ASSETS:
            if s not in FACTORS or s not in DATA:
                continue
            if t not in FACTORS[s].index:
                continue
            idx = DATA[s].index.get_loc(t)
            j = idx + h
            if j >= len(DATA[s]):
                continue
            x = FACTORS[s].loc[t, fname]
            y = DATA[s]["close"].iloc[j] / DATA[s]["close"].iloc[idx] - 1.0
            if pd.notna(x) and pd.notna(y):
                xs.append(x); ys.append(y)
        if len(xs) >= MIN_ASSETS:
            ics[t] = spearman(xs, ys)
    return pd.Series(ics)

def turnover_10d(fname):
    diffs = []
    prev = None
    for t in GRID:
        vals = {s: FACTORS[s].loc[t, fname] for s in ASSETS
                if s in FACTORS and t in FACTORS[s].index and pd.notna(FACTORS[s].loc[t, fname])}
        if len(vals) < MIN_ASSETS:
            prev = None
            continue
        rk = pd.Series(vals).rank(pct=True)
        if prev is not None and len(prev.index.intersection(rk.index)) >= MIN_ASSETS:
            common = prev.index.intersection(rk.index)
            diffs.append(float((rk[common] - prev[common]).abs().mean()))
        prev = rk
    return float(np.mean(diffs)) if diffs else np.nan

def load_library_artifacts():
    import glob, os
    mats = {}
    for f in glob.glob("factors/*.signal.npy"):
        base = os.path.basename(f)[: -len(".signal.npy")]
        mats[base] = np.load(f, allow_pickle=True).astype(float)
    for f in glob.glob("factors/*.json"):
        base = os.path.basename(f)[: -len(".json")]
        try:
            d = json.load(open(f))
        except Exception:
            continue
        art = d.get("signal_artifact")
        if art is None or isinstance(art, str):
            continue
        if isinstance(art, dict):
            vals = art.get("values")
        else:
            vals = art
        if not vals:
            continue
        mats[base] = np.array([[float(v) if v is not None else np.nan for v in row] for row in vals])
    return mats

def max_lib_corr(sigmat, mats):
    best = 0.0
    best_name = None
    for name, m in mats.items():
        if m.shape != sigmat.shape:
            continue
        mask = ~(np.isnan(sigmat) | np.isnan(m))
        if mask.sum() < 500:
            continue
        a = sigmat[mask]; b = m[mask]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        if abs(rho) > best:
            best = abs(rho)
            best_name = name
    return best, best_name

MATS = load_library_artifacts()
print("library artifacts for corr audit:", len(MATS))

print("\n=== SCREENING RESULTS (h=10 Spearman daily IC; gates |IC|>=0.0070 |ICIR|>=0.0840) ===")
summary = {}
for nm in names:
    ic = ic_series(nm).dropna()
    if len(ic) < 200:
        print(f"{nm:24s} insufficient IC dates ({len(ic)})")
        continue
    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    hit = float((ic > 0).mean()) if mean_ic >= 0 else float((ic < 0).mean())
    tot = 0; valid = 0; dates_ge8 = 0
    for t in GRID:
        cnt = 0
        for s in ASSETS:
            if s in FACTORS and t in FACTORS[s].index:
                tot += 1
                v = FACTORS[s].loc[t, nm]
                if pd.notna(v):
                    valid += 1; cnt += 1
        if cnt >= MIN_ASSETS:
            dates_ge8 += 1
    cov = valid / tot if tot else np.nan
    covd = dates_ge8 / len(GRID)
    to = turnover_10d(nm)
    decay = {h: round(float(ic_at_horizon(nm, h).mean()), 4) for h in [1, 2, 3, 5, 10, 20]}
    sigmat = np.full((len(GRID), len(ASSETS)), np.nan)
    for i, t in enumerate(GRID):
        for j, s in enumerate(ASSETS):
            if s in FACTORS and t in FACTORS[s].index:
                v = FACTORS[s].loc[t, nm]
                if pd.notna(v):
                    sigmat[i, j] = v
    mcorr, mname = max_lib_corr(sigmat, MATS)
    regs = {}
    for lo, hi, lbl in [("2020-01-01", "2021-12-31", "2020-21"), ("2022-01-01", "2022-12-31", "2022"),
                        ("2023-01-01", "2024-12-31", "2023-24"), ("2025-01-01", "2099-12-31", "2025-26")]:
        sub = ic[(ic.index >= lo) & (ic.index <= hi)]
        if len(sub) > 50:
            regs[lbl] = (round(float(sub.mean()), 4),
                         round(float(sub.mean() / sub.std()), 3) if sub.std() > 1e-12 else 0.0, len(sub))
    pass_gate = abs(mean_ic) >= 0.0070 and abs(icir) >= 0.0840
    summary[nm] = dict(ic=mean_ic, icir=icir, hit=hit, n=len(ic), cov=cov, covd=covd,
                       to=to, decay=decay, mcorr=mcorr, mname=mname, regs=regs, pass_gate=pass_gate)
    print(f"\n{nm:24s} IC={mean_ic:+.4f} ICIR={icir:+.3f} hit={hit:.3f} n={len(ic)} cov={cov:.3f} covd8={covd:.3f} to10={to:.3f} maxLibCorr={mcorr:.3f}({mname}) PASS={pass_gate}")
    print(f"  decay(h): {decay}")
    for lbl, (ri, rir, rn) in regs.items():
        print(f"  {lbl}: ic={ri:+.4f} icir={rir:+.3f} n={rn}")

json.dump({k: {kk: (vv if not isinstance(vv, dict) else {str(a): b for a, b in vv.items()})
               for kk, vv in v.items()} for k, v in summary.items()},
          open("scripts/_miner2_20260813_screen_results.json", "w"), indent=1, default=str)
print("\nsaved: scripts/_miner2_20260813_screen_results.json")
