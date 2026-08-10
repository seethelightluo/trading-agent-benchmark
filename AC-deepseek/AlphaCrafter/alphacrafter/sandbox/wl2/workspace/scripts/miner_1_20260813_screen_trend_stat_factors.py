"""miner_1 screening: statistical/trend-quality factor family on the 15-asset tradable cross-asset universe.

Candidates (all computed with data visible at date t, no lookahead):
  A. trend_r2_60      : R^2 of OLS log-price linear trend over 60d (trend strength/quality)
  B. autocorr_20      : lag-1 autocorrelation of daily returns over 20d (trend persistence)
  C. skew_60          : Fisher skewness of daily returns over 60d (return path asymmetry)
  D. bollinger_pos_20 : (close - SMA20) / (2*std20) (mean-reversion position)
  E. drawdown_60      : close/rolling_max(close,60) - 1 (continuous distance-from-high)
  F. vol_trend_20x60  : mean(volume,5)/mean(volume,60) - 1 (volume expansion)
  G. range_20         : mean((high-low)/close, 20) (intraday amplitude)
  H. gap_level_20     : mean(open/prev_close - 1, 20) (overnight gap persistence)
  K. kurt_60          : excess kurtosis of daily returns over 60d

Metric: daily cross-sectional Spearman IC vs h=10 forward return, ICIR=mean/std,
hit ratio, coverage, turnover (10d rank change), decay by horizon, max abs library
correlation vs existing effective factor artifacts. Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ART_START = "2020-01-01"
assert ART_START in TRADING_DAYS and VISIBLE in TRADING_DAYS
ROW0 = TRADING_DAYS.index(ART_START)
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
print(f"grid rows: {len(GRID)}  {GRID[0]}..{GRID[-1]}")

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
print("assets:", len(ASSETS))

HORIZON = 10
MIN_ASSETS = 8


def load_asset(sym):
    df = get_stock_daily_data(sym, days=2000)
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
    print(f"  {s:10s} rows={0 if df is None else len(df)} first={0 if df is None else df.index[0]} last={0 if df is None else df.index[-1]}")


def safe_div(a, b):
    return a / np.where(np.abs(b) < 1e-12, np.nan, b)


def compute_factors(df):
    """Return DataFrame (index=date) of candidate factor values for one asset."""
    if df is None or len(df) < 80:
        return None
    close = df["close"]
    ret = close.pct_change()
    out = pd.DataFrame(index=df.index)

    # A. trend R2 over 60d (OLS of log price on time)
    lp = np.log(close)
    x = np.arange(len(lp))
    xm = x - x.mean()
    def r2_window(i, w=60):
        lo = max(0, i - w + 1)
        seg = lp.iloc[lo:i + 1].values
        if len(seg) < 30:
            return np.nan
        xx = xm[i - len(seg) + 1:i + 1]
        yy = seg - seg.mean()
        den = (xx * xx).sum()
        if den < 1e-12 or (yy * yy).sum() < 1e-12:
            return np.nan
        beta = (xx * yy).sum() / den
        resid = yy - beta * xx
        ss_res = (resid * resid).sum()
        ss_tot = (yy * yy).sum()
        return 1.0 - ss_res / ss_tot
    out["trend_r2_60"] = [r2_window(i, 60) for i in range(len(lp))]

    # B. lag-1 autocorr of daily returns over 20d
    def acf20(i):
        seg = ret.iloc[max(0, i - 19):i + 1].values
        if len(seg) < 12:
            return np.nan
        a, b = seg[:-1], seg[1:]
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    out["autocorr_20"] = [acf20(i) for i in range(len(ret))]

    # C. Fisher skewness over 60d
    out["skew_60"] = ret.rolling(60, min_periods=30).skew()

    # D. bollinger position 20
    ma20 = close.rolling(20, min_periods=10).mean()
    sd20 = close.rolling(20, min_periods=10).std()
    out["bollinger_pos_20"] = safe_div(close - ma20, 2 * sd20)

    # E. drawdown 60 (close/rolling_max - 1)
    out["drawdown_60"] = close / close.rolling(60, min_periods=30).max() - 1.0

    # F. volume trend 5/60
    v5 = df["volume"].rolling(5, min_periods=3).mean()
    v60 = df["volume"].rolling(60, min_periods=20).mean()
    out["vol_trend_20x60"] = safe_div(v5, v60) - 1.0

    # G. mean intraday range over 20d
    rng = safe_div(df["high"] - df["low"], close)
    out["range_20"] = rng.rolling(20, min_periods=10).mean()

    # H. overnight gap level over 20d
    gap = safe_div(df["open"], close.shift(1)) - 1.0
    out["gap_level_20"] = gap.rolling(20, min_periods=10).mean()

    # K. excess kurtosis over 60d
    out["kurt_60"] = ret.rolling(60, min_periods=30).kurt()

    return out


FACTORS = {}
for s in ASSETS:
    f = compute_factors(DATA[s])
    if f is not None:
        FACTORS[s] = f
print("factor frames computed for", len(FACTORS), "assets")

# forward returns per asset on own calendar
FWD = {}
for s, df in DATA.items():
    if df is None:
        continue
    close = df["close"]
    fwd = close.shift(-HORIZON) / close - 1.0
    FWD[s] = fwd

names = ["trend_r2_60", "autocorr_20", "skew_60", "bollinger_pos_20",
         "drawdown_60", "vol_trend_20x60", "range_20", "gap_level_20", "kurt_60"]


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
    """Daily cross-sectional Spearman IC series (h=10) on the common grid."""
    ics = {}
    for t in GRID:
        xs, ys = [], []
        for s in ASSETS:
            if s not in FACTORS or s not in FWD:
                continue
            if t not in FACTORS[s].index or t not in FWD[s].index:
                continue
            x = FACTORS[s].loc[t, fname]
            y = FWD[s].loc[t]
            if pd.notna(x) and pd.notna(y):
                xs.append(x)
                ys.append(y)
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
            f = FACTORS[s]
            d = DATA[s]
            if t not in f.index:
                continue
            idx = d.index.get_loc(t)
            j = idx + h
            if j >= len(d):
                continue
            x = f.loc[t, fname]
            y = d["close"].iloc[j] / d["close"].iloc[idx] - 1.0
            if pd.notna(x) and pd.notna(y):
                xs.append(x)
                ys.append(y)
        if len(xs) >= MIN_ASSETS:
            ics[t] = spearman(xs, ys)
    return pd.Series(ics)


def turnover_10d(fname):
    """Mean absolute change of cross-sectional normalized rank over 10d."""
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


# ---- library artifacts (existing effective factors) for correlation audit ----
def load_library_artifacts():
    import glob, os
    mats = {}
    # npy artifacts
    for f in glob.glob("factors/*.signal.npy"):
        base = os.path.basename(f)[: -len(".signal.npy")]
        if base.endswith(".bak"):
            continue
        mats[base] = np.load(f, allow_pickle=True).astype(float)
    # embedded artifacts in json
    for f in glob.glob("factors/*.json"):
        base = os.path.basename(f)[: -len(".json")]
        if "." in base and not base.split(".")[0].endswith("_"):
            pass
        try:
            d = json.load(open(f))
        except Exception:
            continue
        art = d.get("signal_artifact")
        if not isinstance(art, dict):
            continue
        vals = art.get("values")
        if not vals:
            continue
        mats[base] = np.array([[float(v) if v is not None else np.nan for v in row] for row in vals])
    return mats


def max_lib_corr(sigmat, mats):
    best = 0.0
    for name, m in mats.items():
        if m.shape != sigmat.shape:
            continue
        mask = ~(np.isnan(sigmat) | np.isnan(m))
        if mask.sum() < 500:
            continue
        a = sigmat[mask]
        b = m[mask]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        if abs(rho) > best:
            best = abs(rho)
    return best


MATS = load_library_artifacts()
print("library artifacts for corr audit:", len(MATS))

print("\n=== SCREENING RESULTS (h=10 Spearman daily IC, gates |IC|>=0.0070 |ICIR|>=0.0840) ===")
summary = {}
for nm in names:
    ic = ic_series(nm)
    ic = ic.dropna()
    if len(ic) < 200:
        print(f"{nm:20s} insufficient IC dates ({len(ic)})")
        continue
    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    hit = float((ic > 0).mean()) if mean_ic >= 0 else float((ic < 0).mean())
    # coverage
    tot = 0
    valid = 0
    dates_ge8 = 0
    for t in GRID:
        cnt = 0
        for s in ASSETS:
            if s in FACTORS and t in FACTORS[s].index:
                tot += 1
                v = FACTORS[s].loc[t, nm]
                if pd.notna(v):
                    valid += 1
                    cnt += 1
        if cnt >= MIN_ASSETS:
            dates_ge8 += 1
    cov = valid / tot if tot else np.nan
    covd = dates_ge8 / len(GRID)
    to = turnover_10d(nm)
    # decay
    decay = {h: round(float(ic_at_horizon(nm, h).mean()), 4) for h in [1, 2, 3, 5, 10, 20]}
    # library corr (build grid matrix)
    sigmat = np.full((len(GRID), len(ASSETS)), np.nan)
    for i, t in enumerate(GRID):
        for j, s in enumerate(ASSETS):
            if s in FACTORS and t in FACTORS[s].index:
                v = FACTORS[s].loc[t, nm]
                if pd.notna(v):
                    sigmat[i, j] = v
    mcorr = max_lib_corr(sigmat, MATS)
    # regime splits
    regs = {}
    for lo, hi, lbl in [("2020-01-01", "2021-12-31", "2020-21"), ("2022-01-01", "2022-12-31", "2022"),
                        ("2023-01-01", "2024-12-31", "2023-24"), ("2025-01-01", "2099-12-31", "2025-26")]:
        sub = ic[(ic.index >= lo) & (ic.index <= hi)]
        if len(sub) > 50:
            regs[lbl] = (round(float(sub.mean()), 4), round(float(sub.mean() / sub.std()), 3) if sub.std() > 1e-12 else 0.0, len(sub))
    summary[nm] = dict(ic=mean_ic, icir=icir, hit=hit, n=len(ic), cov=cov, covd=covd,
                       to=to, decay=decay, mcorr=mcorr, regs=regs)
    print(f"\n{nm:20s} IC={mean_ic:+.4f} ICIR={icir:+.3f} hit={hit:.3f} n={len(ic)} cov={cov:.3f} covd8={covd:.3f} to10={to:.3f} maxLibCorr={mcorr:.3f}")
    print(f"  decay(h): {decay}")
    for lbl, (ri, rir, rn) in regs.items():
        print(f"  {lbl}: ic={ri:+.4f} icir={rir:+.3f} n={rn}")

json.dump({k: {kk: (vv if not isinstance(vv, dict) else {str(a): b for a, b in vv.items()})
               for kk, vv in v.items()} for k, v in summary.items()},
          open("scripts/miner_1_20260813_screen_results.json", "w"), indent=1, default=str)
print("\nsaved: scripts/miner_1_20260813_screen_results.json")
