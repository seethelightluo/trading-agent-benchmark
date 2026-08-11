"""
miner_2 cycle: screen NEW factor families for diversity vs the reversal-heavy library.
Families: liquidity (amihud, volume z), volatility shape (skew, range ratio),
intraday shape (upper/lower shadow), trend efficiency, medium momentum, macro beta (US10Y, USDCNY).
Validation window 2020-01-02..2026-07-15 (warm-up), gates |IC|>=0.0070, |ICIR|>=0.0840 at 10d horizon,
plus decay at 1/2/3/5d and max abs library correlation.
"""
import pickle, time, numpy as np, pandas as pd

T0 = time.time()
SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
IC_MIN, ICIR_MIN = 0.0070, 0.0840
MIN_NAMES = 8

cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close = cache["close"][SYMBOLS]; open_ = cache["open"][SYMBOLS]
high = cache["high"][SYMBOLS]; low = cache["low"][SYMBOLS]
vol = cache["vol"][SYMBOLS]
idx = close.index
ret = np.log(close).diff()
macro = cache["macro"]

def fwd_log(closes, h):
    return np.log(closes.shift(-h)) - np.log(closes)

def fast_ic(F, R, min_names=MIN_NAMES):
    F = F.values.astype(float); R = R.values.astype(float)
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

def per_symbol_dense(fn):
    out = pd.DataFrame(np.nan, index=idx, columns=SYMBOLS)
    for c in SYMBOLS:
        s = close[c].dropna()
        if len(s) < 30:
            continue
        out[c] = fn(s)
    return out

def rstd(s, w): return s.pct_change().rolling(w).std()
def rmean(s, w): return s.pct_change().rolling(w).mean()

# ---------------- candidate factors (dense per symbol) ----------------
cands = {}

# 1. Amihud illiquidity 20d: mean(|ret| / volume) -- higher = more illiquid
def _amihud(s, w=20):
    r = s.pct_change().abs()
    v = vol[s.name]
    a = (r / v.replace(0, np.nan)).rolling(w).mean()
    return -np.log1p(a)  # negate: predict return (liquidity premium -> illiquidity up, ret up?)
cands["amihud_illiq_20"] = per_symbol_dense(_amihud)

# 2. Volume z-score 10d vs 60d: attention/surge
def _volz(s):
    v = vol[s.name].rolling(10).mean()
    vv = vol[s.name].rolling(60).mean()
    return v / vv - 1.0
cands["volume_z_10_60"] = per_symbol_dense(_volz)

# 3. 30d return skewness (lottery)
def _skew(s, w=30):
    return s.pct_change().rolling(w).skew()
cands["skew_30"] = per_symbol_dense(_skew)

# 4. Range ratio 3/20: short-term vol regime vs long
def _rr(s):
    r3 = s.pct_change().rolling(3).std()
    r20 = s.pct_change().rolling(20).std()
    return r3 / r20 - 1.0
cands["range_ratio_3_20"] = per_symbol_dense(_rr)

# 5. Upper shadow 5d: mean((high - max(o,c))/close)
def _ush(s, w=5):
    o = open_[s.name].reindex(s.index); h = high[s.name].reindex(s.index)
    sh = (h - np.maximum(o, s)) / s
    return -sh.rolling(w).mean()  # negate: more upper shadow -> lower ret
cands["upper_shadow_5"] = per_symbol_dense(_ush)

# 6. Lower shadow 5d: mean((min(o,c) - low)/close)
def _lsh(s, w=5):
    o = open_[s.name].reindex(s.index); l = low[s.name].reindex(s.index)
    sh = (np.minimum(o, s) - l) / s
    return sh.rolling(w).mean()
cands["lower_shadow_5"] = per_symbol_dense(_lsh)

# 7. Trend efficiency 60d: |net move| / path length
def _tef(s, w=60):
    disp = (s / s.shift(w) - 1.0).abs()
    path = s.pct_change().abs().rolling(w).sum()
    return disp / path
cands["trend_eff_60"] = per_symbol_dense(_tef)

# 8. Medium momentum 20d skip5
def _mom20(s):
    return np.log(s.shift(5) / s.shift(25))
cands["mom_20d_skip5"] = per_symbol_dense(_mom20)

# 9. US10Y beta 60d (rate sensitivity)
def _us10y_beta(s):
    u = np.log(close["US10Y"]).diff().reindex(s.index)
    a = s.pct_change(); b = u
    mu = a.rolling(60).mean(); mb = b.rolling(60).mean()
    cov = ((a - mu) * (b - mb)).rolling(60).mean()
    var = ((b - mb) ** 2).rolling(60).mean()
    return cov / var
cands["us10y_beta_60"] = per_symbol_dense(_us10y_beta)

# 10. USDCNY beta 60d (EM/CN risk sensitivity)
def _cny_beta(s):
    u = np.log(macro["USDCNY"]).diff().reindex(s.index)
    a = s.pct_change(); b = u
    mu = a.rolling(60).mean(); mb = b.rolling(60).mean()
    cov = ((a - mu) * (b - mb)).rolling(60).mean()
    var = ((b - mb) ** 2).rolling(60).mean()
    return cov / var
cands["usdcny_beta_60"] = per_symbol_dense(_cny_beta)

# 11. Negative 5d momentum (short reversal at 5d, vol-scaled variant)
def _rev5_vs(s):
    r5 = np.log(s) - np.log(s.shift(5))
    v = s.pct_change().rolling(20).std()
    return -(r5 / v)
cands["rev_5d_volscaled"] = per_symbol_dense(_rev5_vs)

# ---------------- library signals for correlation ----------------
def build_lib_signals():
    lib = {}
    def neg(x): return -x
    lib["rev_1d"] = -ret
    lib["rev_2d"] = -(np.log(close) - np.log(close.shift(2)))
    lib["rev_3d"] = -(np.log(close) - np.log(close.shift(3)))
    lib["rev_5d"] = -(np.log(close) - np.log(close.shift(5)))
    lib["rev_1d_vs"] = -(np.log(close) - np.log(close.shift(1))) / ret.rolling(20).std()
    lib["id_rev_1d"] = -(close / open_ - 1.0)
    rng_1 = (high.rolling(1).max() - low.rolling(1).min())
    lib["nclv_1d"] = -(close - low.rolling(1).min()) / rng_1
    rng_2 = (high.rolling(2).max() - low.rolling(2).min())
    lib["nclv_2d"] = -(close - low.rolling(2).min()) / rng_2
    rng_3 = (high.rolling(3).max() - low.rolling(3).min())
    lib["nclv_3d"] = -(close - low.rolling(3).min()) / rng_3
    rng_5 = (high.rolling(5).max() - low.rolling(5).min())
    lib["nclv_5d"] = -(close - low.rolling(5).min()) / rng_5
    body = (close - open_) / (high - low)
    lib["nbody_1d"] = -body
    lib["mom_10d_skip5"] = np.log(close.shift(5) / close.shift(15))
    lib["mom_120d_skip5"] = np.log(close.shift(5) / close.shift(125))
    vix = macro["VIX"]
    vixr = np.log(vix).diff().reindex(idx)
    def vbeta(s):
        a = s.pct_change(); b = vixr
        mu = a.rolling(60).mean(); mb = b.rolling(60).mean()
        cov = ((a - mu) * (b - mb)).rolling(60).mean()
        var = ((b - mb) ** 2).rolling(60).mean()
        return cov / var
    lib["vix_beta_cond_60x20"] = -vbeta(close) * (vix.reindex(idx) / vix.reindex(idx).shift(20) - 1.0)
    lib["vol_of_vol20x60"] = ret.rolling(20).std().rolling(60).std()
    return lib

def pair_rho(a, b):
    A = a.values.astype(float); B = b.values.astype(float)
    vals = []
    for i in range(len(A)):
        x, y = A[i], B[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < MIN_NAMES:
            continue
        x, y = x[m], y[m]
        rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
        if rx.std() <= 1e-12 or ry.std() <= 1e-12:
            continue
        vals.append(abs(float(np.corrcoef(rx, ry)[0, 1])))
    return float(np.mean(vals)) if vals else np.nan

lib = build_lib_signals()
print("library signals built:", list(lib.keys()))

# ---------------- validation ----------------
print("\n=== CANDIDATE SCREEN (2020-01-02..2026-07-15) ===")
hdr = f"{'factor':<22}{'ic1':>7}{'ic2':>7}{'ic3':>7}{'ic5':>7}{'ic10':>8}{'icir10':>8}{'hit10':>7}{'cov':>6}{'to10':>7}{'rho':>6}  gate"
print(hdr)
results = {}
for name, F in cands.items():
    cov = float(F.notna().mean().mean())
    to = turnover10(F)
    row = {"factor": name, "cov": cov, "to10": to}
    for h in [1, 2, 3, 5, 10]:
        r = fast_ic(F, fwd_log(close, h))
        row[f"ic{h}"] = r["ic"]; row[f"n{h}"] = r["n_dates"]
        if h == 10:
            row["icir10"] = r["icir"]; row["hit10"] = r["hit"]; row["n10"] = r["n_dates"]
    rhos = [pair_rho(F, lib[k]) for k in lib]
    row["max_rho"] = float(np.nanmax(rhos))
    results[name] = row
    print(f"{name:<22}{row['ic1']:>7.3f}{row['ic2']:>7.3f}{row['ic3']:>7.3f}{row['ic5']:>7.3f}"
          f"{row['ic10']:>8.3f}{row['icir10']:>8.3f}{row['hit10']:>7.3f}{row['cov']:>6.2f}"
          f"{row['to10']:>7.2f}{row['max_rho']:>6.2f}  {'PASS' if abs(row['ic10'])>=IC_MIN and abs(row['icir10'])>=ICIR_MIN else ''}")

print("\nelapsed %.1fs" % (time.time() - T0))
