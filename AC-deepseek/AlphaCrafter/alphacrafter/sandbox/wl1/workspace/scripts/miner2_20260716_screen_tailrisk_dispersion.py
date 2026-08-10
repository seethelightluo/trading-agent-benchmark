"""
miner_2 cycle 3: tail-risk / liquidity / cross-asset-beta families.

Motivation: the effective library is dominated by short-horizon mean-reversion
(nclv_1d, rev_1d, rev_intraday_1d, clv_5d, mom_10d_skip5-as-reversal). To add
diversified alpha we explore factor families that capture DIFFERENT return
drivers: return skewness/kurtosis (tail risk), downside-vol concentration,
autocorrelation persistence, MAX-effect reversal, intraday body/gap structure,
volume-confirmed signals, and cross-asset beta exposures (BTC/XAU as global
risk factors). New entrants must pass the shared admission gate
(|daily IC| >= 0.007, |ICIR| >= 0.084) and stay pairwise-uncorrelated
(|rho| < 0.5) with the persisted library.

All factors are computed per-symbol on each asset's own trading days (dense),
then reindexed to the calendar panel; IC is a per-date cross-sectional rank
correlation with forward 1-day log returns over 2021-01-01..2026-07-15
(>= 8 valid names per date).
"""
import sys, os, json, pickle, time, base64, zlib, io
import numpy as np
import pandas as pd

T0 = time.time()
SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
VALID_LO, VALID_HI = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-07-15")
IC_MIN, ICIR_MIN = 0.0070, 0.0840
RHO_MAX = 0.5
MIN_NAMES = 8

cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close = cache["close"][SYMBOLS]; open_ = cache["open"][SYMBOLS]
high = cache["high"][SYMBOLS]; low = cache["low"][SYMBOLS]
vol = cache["vol"][SYMBOLS]
idx = close.index
lret = np.log(close).diff()

# -----------------------------------------------------------------------------
# fast cross-sectional rank IC
# -----------------------------------------------------------------------------
def fwd_log(closes, h):
    return np.log(closes.shift(-h)) - np.log(closes)

def fast_ic(factor_df, fwd, min_names=MIN_NAMES):
    F = factor_df.values.astype(float); R = fwd.values.astype(float)
    n = np.isfinite(F) & np.isfinite(R)
    ok = n.sum(axis=1) >= min_names
    if not ok.any():
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    Fm = np.where(n, F, 0.0); Rm = np.where(n, R, 0.0)
    cnt = n.sum(axis=1)[ok]
    sx = Fm[ok].sum(axis=1); sy = Rm[ok].sum(axis=1)
    sxx = (Fm[ok] ** 2).sum(axis=1); syy = (Rm[ok] ** 2).sum(axis=1)
    sxy = (Fm[ok] * Rm[ok]).sum(axis=1)
    with np.errstate(all="ignore"):
        num = cnt * sxy - sx * sy
        den = np.sqrt((cnt * sxx - sx * sx) * (cnt * syy - sy * sy))
        ic = num / den
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    return {"n_dates": int(len(ic)), "n_obs": int(cnt.sum()),
            "ic": float(ic.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
            "hit": float((ic > 0).mean())}

def turnover10(factor_df, rebal=10):
    ranks = factor_df.rank(axis=1)
    chg = []
    for i in range(rebal, len(ranks)):
        prev = ranks.iloc[i - rebal].dropna(); cur = ranks.iloc[i].dropna()
        common = prev.index.intersection(cur.index)
        if len(common) < 2:
            continue
        chg.append((cur[common] - prev[common]).abs().mean() / (len(common) - 1))
    return float(np.mean(chg)) if chg else np.nan

def pair_rho(a, b):
    A = a.values.astype(float); B = b.values.astype(float)
    vals = []
    for i in range(len(A)):
        x, y = A[i], B[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 8:
            continue
        x, y = x[m], y[m]
        rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
        if rx.std() <= 1e-12 or ry.std() <= 1e-12:
            continue
        vals.append(abs(float(np.corrcoef(rx, ry)[0, 1])))
    return float(np.mean(vals)) if vals else np.nan

def per_symbol_dense(fn, extra=None):
    out = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
    for c in SYMBOLS:
        s = close[c].dropna()
        if len(s) < 90:
            continue
        out[c] = fn(s, c) if extra else fn(s)
    return out

# -----------------------------------------------------------------------------
# build candidate panels
# -----------------------------------------------------------------------------
def roll_skew(s, w): return s.pct_change().rolling(w).skew()
def roll_kurt(s, w): return s.pct_change().rolling(w).kurt()
def roll_autocorr(s, w):
    r = s.pct_change()
    return r.rolling(w).apply(lambda x: pd.Series(x).autocorr(1) if len(x) >= 8 else np.nan, raw=False)

panels = {}
META = {}

# --- tail risk: negative skewness ---
panels["skew_neg_20"] = -per_symbol_dense(lambda s: roll_skew(s, 20))
META["skew_neg_20"] = {"name": "Negative 20d return skewness", "expr": "-skew(pct_ret, 20)",
                       "dep": ["close"], "params": {"win": 20}, "tags": ["tail-risk", "skewness"]}

panels["skew_neg_60"] = -per_symbol_dense(lambda s: roll_skew(s, 60))
META["skew_neg_60"] = {"name": "Negative 60d return skewness", "expr": "-skew(pct_ret, 60)",
                       "dep": ["close"], "params": {"win": 60}, "tags": ["tail-risk", "skewness"]}

# --- tail risk: negative kurtosis ---
panels["kurt_neg_20"] = -per_symbol_dense(lambda s: roll_kurt(s, 20))
META["kurt_neg_20"] = {"name": "Negative 20d return kurtosis", "expr": "-kurt(pct_ret, 20)",
                       "dep": ["close"], "params": {"win": 20}, "tags": ["tail-risk", "kurtosis"]}

# --- autocorrelation persistence ---
panels["autocorr_neg_10"] = -per_symbol_dense(lambda s: roll_autocorr(s, 10))
META["autocorr_neg_10"] = {"name": "Negative 10d return autocorr lag1", "expr": "-autocorr(pct_ret, lag=1, win=10)",
                           "dep": ["close"], "params": {"win": 10, "lag": 1}, "tags": ["autocorrelation"]}

# --- downside vol concentration ---
def downside_ratio(s, w=20):
    r = s.pct_change()
    return r.rolling(w).apply(lambda x: (np.nanstd(x[x < 0]) / np.nanstd(x)) if (x < 0).any() and np.nanstd(x) > 0 else np.nan, raw=True)
panels["downside_vol_20"] = per_symbol_dense(lambda s: downside_ratio(s, 20))
META["downside_vol_20"] = {"name": "Downside vol ratio 20d", "expr": "std(ret|ret<0)/std(ret) over 20d",
                           "dep": ["close"], "params": {"win": 20}, "tags": ["tail-risk", "downside-vol"]}

# --- MAX effect ---
panels["max_neg_20"] = -per_symbol_dense(lambda s: s.pct_change().rolling(20).max())
META["max_neg_20"] = {"name": "Negative 20d max daily return (MAX effect)", "expr": "-max(pct_ret, 20)",
                      "dep": ["close"], "params": {"win": 20}, "tags": ["tail-risk", "lottery"]}

# --- intraday body position (open-based CLV) ---
_p = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    df = pd.concat([open_[c], close[c], high[c], low[c]], axis=1).dropna()
    if len(df) < 30:
        continue
    rng = (df.iloc[:, 2] - df.iloc[:, 3])
    body = (df.iloc[:, 1] - df.iloc[:, 0]) / rng.replace(0, np.nan)
    _p.loc[df.index, c] = body
panels["body_pos_1d"] = _p
META["body_pos_1d"] = {"name": "Intraday body position (close-open)/(high-low)", "expr": "(close-open)/(high-low)",
                       "dep": ["open", "close", "high", "low"], "params": {}, "tags": ["intraday", "range"]}

# --- overnight gap ---
_p = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    df = pd.concat([open_[c], close[c]], axis=1).dropna()
    if len(df) < 30:
        continue
    gap = (df.iloc[:, 0] - df.iloc[:, 1].shift(1)) / df.iloc[:, 1].shift(1)
    _p.loc[df.index, c] = gap
panels["gap_1d"] = _p
META["gap_1d"] = {"name": "Overnight gap 1d", "expr": "(open - prev_close)/prev_close",
                  "dep": ["open", "close"], "params": {}, "tags": ["intraday", "gap"]}

# --- volume z-score (attention/liquidity) ---
def vol_z(sym, w=20, wl=60):
    v = vol[sym]
    mu = v.rolling(wl).mean(); sd = v.rolling(wl).std()
    return (v - mu) / sd
panels["volz_5_60"] = per_symbol_dense(lambda s, c: vol_z(c, 5, 60), extra=True)
META["volz_5_60"] = {"name": "Volume z-score 5d vs 60d", "expr": "(vol - mean(vol,60))/std(vol,60)",
                     "dep": ["vol"], "params": {"win_s": 5, "win_l": 60}, "tags": ["liquidity", "volume"]}

# --- volume-confirmed reversal: -ret_1d * volz (high-volume days revert harder) ---
_p = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
_z = panels["volz_5_60"]
r1 = lret
_p = -r1 * _z
panels["vol_confirmed_rev"] = _p
META["vol_confirmed_rev"] = {"name": "Volume-confirmed 1d reversal", "expr": "-ret_1d * zscore(vol,60)",
                             "dep": ["close", "vol"], "params": {}, "tags": ["reversal", "volume"]}

# --- Amihud illiquidity (negated) ---
_p = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    df = pd.concat([lret[c], vol[c]], axis=1).dropna()
    if len(df) < 60 or df.iloc[:, 1].abs().sum() == 0:
        continue
    amihud = (df.iloc[:, 0].abs() / df.iloc[:, 1]).replace([np.inf, -np.inf], np.nan)
    _p.loc[df.index, c] = -amihud.rolling(20).mean()
panels["amihud_neg_20"] = _p
META["amihud_neg_20"] = {"name": "Negative Amihud illiquidity 20d", "expr": "-mean(|ret|/vol, 20)",
                         "dep": ["close", "vol"], "params": {"win": 20}, "tags": ["liquidity"]}

# --- cross-asset beta: BTC as global risk factor ---
rbtc = lret["BTC"]
def beta_btc(s, w=60):
    r = s.pct_change()
    df = pd.concat([r, rbtc], axis=1).dropna()
    if len(df) < w + 10:
        return pd.Series(np.nan, index=s.index)
    cov = df.iloc[:, 0].rolling(w).cov(df.iloc[:, 1])
    var = df.iloc[:, 1].rolling(w).var()
    return (cov / var).reindex(s.index)
panels["btc_beta_60"] = per_symbol_dense(lambda s: beta_btc(s, 60))
META["btc_beta_60"] = {"name": "60d beta vs BTC returns", "expr": "beta(ret_i, ret_BTC, 60)",
                       "dep": ["close"], "params": {"win": 60}, "tags": ["cross-asset-beta", "crypto"]}

# --- cross-asset beta: XAU as safe-haven factor ---
rxau = lret["XAU"]
def beta_xau(s, w=60):
    r = s.pct_change()
    df = pd.concat([r, rxau], axis=1).dropna()
    if len(df) < w + 10:
        return pd.Series(np.nan, index=s.index)
    cov = df.iloc[:, 0].rolling(w).cov(df.iloc[:, 1])
    var = df.iloc[:, 1].rolling(w).var()
    return (cov / var).reindex(s.index)
panels["xau_beta_60"] = per_symbol_dense(lambda s: beta_xau(s, 60))
META["xau_beta_60"] = {"name": "60d beta vs XAU returns", "expr": "beta(ret_i, ret_XAU, 60)",
                       "dep": ["close"], "params": {"win": 60}, "tags": ["cross-asset-beta", "safe-haven"]}

# --- 5d CLV (distance from recent low) ---
_p = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    df = pd.concat([close[c], high[c], low[c]], axis=1).dropna()
    if len(df) < 30:
        continue
    h5 = df.iloc[:, 1].rolling(5).max(); l5 = df.iloc[:, 2].rolling(5).min()
    rng = (h5 - l5).replace(0, np.nan)
    _p.loc[df.index, c] = (df.iloc[:, 0] - l5) / rng
panels["clv_5d"] = _p
META["clv_5d"] = {"name": "5d close location value", "expr": "(close - min(low,5))/(max(high,5)-min(low,5))",
                  "dep": ["close", "high", "low"], "params": {"win": 5}, "tags": ["mean-reversion", "range"]}

# --- drawdown depth from 60d high ---
_p = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
for c in SYMBOLS:
    s = close[c].dropna()
    if len(s) < 90:
        continue
    hh = s.rolling(60).max()
    _p.loc[s.index, c] = (s - hh) / hh
panels["dd_60"] = _p
META["dd_60"] = {"name": "Drawdown depth from 60d high", "expr": "(close - max(high,60))/max(high,60)",
                 "dep": ["close"], "params": {"win": 60}, "tags": ["drawdown", "trend"]}

print(f"panels built in {time.time()-T0:.1f}s: {list(panels.keys())}")

# -----------------------------------------------------------------------------
# IC evaluation
# -----------------------------------------------------------------------------
m = (idx >= VALID_LO) & (idx <= VALID_HI)
fwd1 = fwd_log(close, 1); fwd5 = fwd_log(close, 5); fwd10 = fwd_log(close, 10)
n_cells = len(SYMBOLS) * int(m.sum())

results = {}
for nm, p in panels.items():
    P = p.loc[m]
    ic1 = fast_ic(P, fwd1.loc[m]); ic5 = fast_ic(P, fwd5.loc[m]); ic10 = fast_ic(P, fwd10.loc[m])
    cov = float(P.notna().sum().sum()) / n_cells
    to = turnover10(p.loc[m])
    passed = (abs(ic1["ic"]) >= IC_MIN) and (abs(ic1["icir"]) >= ICIR_MIN)
    results[nm] = {"panel": p, "ic1": ic1, "ic5": ic5, "ic10": ic10, "cov": cov, "to": to, "passed": passed}
    print(f"{nm:18s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.3f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} | IC10={ic10['ic']:+.4f} | "
          f"{'PASS' if passed else 'fail'}")

# -----------------------------------------------------------------------------
# correlation vs persisted library (load artifacts)
# -----------------------------------------------------------------------------
def load_npy(pid):
    M = np.load(f"factors/{pid}.npy")
    return pd.DataFrame(M, index=idx, columns=SYMBOLS)

def decode_b64zlibcsv(payload):
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    txt = raw.decode("utf-8", errors="replace")
    rows = []
    for line in txt.splitlines():
        if not line.strip() or line.startswith("date,"):
            continue
        parts = line.split(",")
        rows.append((parts[0], [float(v) if v not in ("", "NA") else np.nan for v in parts[1:]]))
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows], columns=payload.get("columns", SYMBOLS))
    df.index = pd.to_datetime(df.index)
    return df

def decode_gzipb64(payload):
    raw = base64.b64decode(payload["data"])
    try:
        raw = zlib.decompress(raw)
    except Exception:
        pass
    arr = np.frombuffer(raw, dtype="<f4").reshape(payload["n_dates"], payload["n_symbols"])
    start = pd.Timestamp(payload["date_start"]); end = pd.Timestamp(payload["date_end"])
    dates = pd.bdate_range(start, end)[: payload["n_dates"]]
    return pd.DataFrame(arr, index=dates, columns=payload.get("symbols", SYMBOLS))

def load_lib_factor(path):
    d = json.load(open(path))
    sa = d.get("signal_artifact")
    if sa is None:
        v = d.get("validation") or {}
        sa = v.get("signal_artifact")
        if sa is None:
            sa = (v.get("metrics") or {}).get("signal_artifact")
    if isinstance(sa, str):
        p = os.path.join("factors", sa)
        if p.endswith(".npy") and os.path.exists(p):
            M = np.load(p)
            return pd.DataFrame(M, index=idx, columns=SYMBOLS)
        return None
    if isinstance(sa, dict):
        fmt = str(sa.get("format", ""))
        if "base64:zlib:csv" in fmt or "data" in sa:
            try:
                return decode_b64zlibcsv(sa)
            except Exception:
                pass
        if "gzip" in fmt and "n_dates" in sa:
            try:
                return decode_gzipb64(sa)
            except Exception:
                pass
    return None

lib_paths = sorted([os.path.join("factors", f) for f in os.listdir("factors")
                    if f.endswith(".json") and not f.startswith("miner2_20260715")])
lib = {}
for p in lib_paths:
    df = load_lib_factor(p)
    if df is not None and len(df) > 100:
        lib[os.path.basename(p)] = df

print(f"\nlibrary loaded: {list(lib.keys())}")
for k, v in lib.items():
    print(f"  {k:45s} shape={v.shape} finite={np.isfinite(v.values).sum()}")

# -----------------------------------------------------------------------------
# selection: pass gate + diverse vs library + mutually diverse
# -----------------------------------------------------------------------------
def quality(nm):
    r = results[nm]
    return abs(r["ic1"]["ic"]) * abs(r["ic1"]["icir"])

passers = [nm for nm, r in results.items() if r["passed"]]
print(f"\npassing gate: {passers}")
for nm in passers:
    row = " ".join(f"{k.split('.')[0][-10:]}:{pair_rho(results[nm]['panel'], v):.2f}" for k, v in lib.items())
    print(f"  {nm:18s} {row}")

kept = []
for nm in sorted(passers, key=lambda x: -quality(x)):
    ok = all(pair_rho(results[nm]["panel"], v) < RHO_MAX for v in lib.values()) and \
         all(pair_rho(results[nm]["panel"], results[k]["panel"]) < RHO_MAX for k in kept)
    if ok:
        kept.append(nm)
print(f"\ndiverse kept: {kept}")
for nm in kept:
    r = results[nm]
    print(f"  {nm:18s} IC1={r['ic1']['ic']:+.4f} ICIR1={r['ic1']['icir']:+.3f} quality={quality(nm):.5f} cov={r['cov']:.3f} to={r['to']:.3f}")

print(f"\nfinished in {time.time()-T0:.1f}s")
