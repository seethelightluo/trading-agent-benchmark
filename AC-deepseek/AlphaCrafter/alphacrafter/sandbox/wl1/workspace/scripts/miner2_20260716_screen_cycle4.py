"""
miner_2 cycle 4 (2026-07-16): screen NOVEL factor families NOT covered by library.
Families: extremes/lottery (MAX), distance-from-low, volatility compression (BB width/NR7),
gap structure, momentum term structure, return asymmetry, volume- and dispersion-conditional
reversal, Hurst persistence, VWAP anchoring, growth-signal beta (COPPER/XAU).
Gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840; keep rho < 0.5 vs library and pairwise.
"""
import sys, os, json, pickle, time, base64, gzip, glob
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
ret = np.log(close).diff()
lvol = np.log(vol.clip(lower=1e-9))
macro = cache["macro"]
vix = macro["VIX"].reindex(idx).ffill()
vixr = np.log(vix).diff()

m = (idx >= VALID_LO) & (idx <= VALID_HI)
fwd1 = np.log(close.shift(-1)) - np.log(close)
fwd5 = np.log(close.shift(-5)) - np.log(close)
fwd10 = np.log(close.shift(-10)) - np.log(close)
n_cells = len(SYMBOLS) * int(m.sum())

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

def load_library_signals():
    signals = {}
    for p in sorted(glob.glob("factors/*.json")):
        if ".bak" in p:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        fid = d.get("factor_id", os.path.basename(p)[:-5])
        if fid in signals:
            continue
        art = d.get("signal_artifact")
        if isinstance(art, dict) and "data_b64" in art:
            raw = base64.b64decode(art["data_b64"])
            arr = np.frombuffer(gzip.decompress(raw), dtype=np.float32)
            arr = arr.reshape(art["n_dates"], art["n_symbols"])
            start = pd.Timestamp(art["date_start"])
            pos = idx.searchsorted(start)
            if pos + arr.shape[0] <= len(idx):
                df = pd.DataFrame(arr, index=idx[pos:pos + arr.shape[0]], columns=SYMBOLS)
            else:
                continue
            signals[fid] = df
        elif isinstance(art, str) and art.endswith(".npy"):
            path = os.path.join("factors", art)
            if not os.path.exists(path):
                continue
            arr = np.load(path)
            if arr.shape[0] == len(idx):
                df = pd.DataFrame(arr, index=idx, columns=SYMBOLS)
            else:
                df = pd.DataFrame(arr, index=idx[:arr.shape[0]], columns=SYMBOLS)
            signals[fid] = df
    return signals

lib = load_library_signals()
print(f"library signals aligned: {len(lib)}")

def pair_rho(a, b):
    A = a.values.astype(float); B = b.values.astype(float)
    vals = []
    for i in range(len(A)):
        x, y = A[i], B[i]
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() < 8:
            continue
        x, y = x[mm], y[mm]
        rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
        if rx.std() <= 1e-12 or ry.std() <= 1e-12:
            continue
        vals.append(abs(float(np.corrcoef(rx, ry)[0, 1])))
    return float(np.mean(vals)) if vals else np.nan

def max_lib_rho(panel):
    best, bestf = 0.0, None
    for fid, sig in lib.items():
        r = pair_rho(panel, sig)
        if np.isfinite(r) and r > best:
            best, bestf = r, fid
    return best, bestf

# ---------------- candidate panels (novel families) ----------------
P = {}
r1 = ret.shift(1)  # prior-day return

# 1. extremes / lottery family (Bali MAX effect, reversal of extremes)
P["neg_max_ret_20"] = -ret.rolling(20).max()          # lottery: past max gain reverts
P["min_ret_20"] = ret.rolling(20).min()                # distress: past worst day
P["extreme_spread_20"] = ret.rolling(20).max() - ret.rolling(20).min()

# 2. distance from low (classic reversal anchor)
P["dist_low_252"] = close / close.rolling(252).min() - 1
P["dist_low_60"] = close / close.rolling(60).min() - 1

# 3. volatility compression
mid20 = close.rolling(20).mean()
sd20 = close.rolling(20).std()
P["bb_bandwidth_20"] = 4 * sd20 / mid20                # Bollinger bandwidth (compression)
rng = high - low
P["nr7_neg"] = -(rng / rng.rolling(7).min())           # narrow-range day (NR7 style)

# 4. gap structure
on_ret = np.log(open_ / close.shift(1))
id_ret = np.log(close / open_)
P["gap_follow_1d"] = on_ret * id_ret                   # gap + intraday follow-through
gap_z20 = (on_ret - on_ret.rolling(20).mean()) / on_ret.rolling(20).std()
P["gap_z_rev_20"] = -gap_z20                           # large gaps revert

# 5. momentum term structure (short vs intermediate)
P["mom_term_5_20"] = np.log(close / close.shift(5)) - np.log(close / close.shift(20))

# 6. return asymmetry
up = ret.clip(lower=0); dn = ret.clip(upper=0)
P["asym_mean_20"] = up.rolling(20).mean() + dn.rolling(20).mean()   # up-mean - |down-mean|
P["up_vol_ratio_20"] = (ret.where(ret > 0).rolling(20).std()
                        / ret.where(ret < 0).rolling(20).std()) - 1

# 7. volume-confirmed reversal
vol_z20 = (lvol - lvol.rolling(20).mean()) / lvol.rolling(20).std()
P["vol_conf_rev_1d"] = -r1 * (1 + vol_z20.clip(lower=0))

# 8. dispersion-conditional reversal (cross-sectional dispersion timing)
disp = ret.std(axis=1)                                # cross-sectional dispersion
disp_z = (disp - disp.rolling(60).mean()) / disp.rolling(60).std()
P["disp_cond_rev_1d"] = -r1 * disp_z.shift(1)         # reversal stronger in high-disp regime

# 9. Hurst persistence (rolling rescaled range, window 20)
def hurst_est(x, min_len=8):
    x = x[~np.isnan(x)]
    if len(x) < min_len:
        return np.nan
    x = x - x.mean()
    cum = np.cumsum(x)
    R = cum.max() - cum.min()
    S = x.std()
    if S <= 1e-12:
        return np.nan
    return R / S

P["hurst_20"] = ret.rolling(20).apply(lambda x: hurst_est(np.asarray(x)), raw=False)

# 10. VWAP anchoring (close vs volume-weighted average price over 20d)
typ = (high + low + close) / 3.0
vwap20 = (typ * vol).rolling(20).sum() / vol.rolling(20).sum()
P["vwap_dist_20"] = close / vwap20 - 1

# 11. growth-signal beta (COPPER/XAU ratio as risk-on growth proxy)
gx = np.log(close["COPPER"] / close["XAU"]).diff()
gx_mom20 = np.log(close["COPPER"] / close["XAU"]).diff(20)
gx_prev = gx.shift(1)
beta_gx60 = ret.rolling(60).cov(gx_prev) / gx_prev.rolling(60).var()
P["growth_beta_60x20"] = beta_gx60 * gx_mom20.shift(1)

# 12. VIX-regime reversal (risk-off flips reversal to trend?)
vix_z = (vix / vix.rolling(252).mean() - 1).shift(1)
P["vix_reg_rev_1d"] = -r1 * (1 + 2 * vix_z.clip(lower=0))

print(f"candidates: {len(P)}")

results = {}
for nm, p in P.items():
    Pp = p.loc[m]
    ic1 = fast_ic(Pp, fwd1.loc[m]); ic5 = fast_ic(Pp, fwd5.loc[m]); ic10 = fast_ic(Pp, fwd10.loc[m])
    cov = float(Pp.notna().sum().sum()) / n_cells
    to = turnover10(Pp)
    passed = (abs(ic1["ic"]) >= IC_MIN) and (abs(ic1["icir"]) >= ICIR_MIN)
    results[nm] = {"panel": p, "ic1": ic1, "ic5": ic5, "ic10": ic10, "cov": cov, "to": to, "passed": passed}
    print(f"{nm:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.3f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} | IC10={ic10['ic']:+.4f} | "
          f"{'PASS' if passed else 'fail'}")

passers = [nm for nm, r in results.items() if r["passed"]]
print(f"\npassing gate: {passers}")

for nm in passers:
    rho, fid = max_lib_rho(results[nm]["panel"].loc[m])
    results[nm]["max_lib_rho"] = rho; results[nm]["max_lib_rho_id"] = fid
    print(f"  {nm:22s} max_lib_rho={rho:.3f} vs {fid}")

def quality(nm):
    r = results[nm]
    return abs(r["ic1"]["ic"]) * abs(r["ic1"]["icir"])

kept = []
for nm in sorted(passers, key=lambda x: -quality(x)):
    r = results[nm]
    if r.get("max_lib_rho", 0) >= RHO_MAX:
        print(f"  drop {nm}: too close to library ({r['max_lib_rho']:.3f})")
        continue
    if all(pair_rho(results[nm]["panel"].loc[m], results[k]["panel"].loc[m]) < RHO_MAX for k in kept):
        kept.append(nm)
print(f"\ndiverse kept: {kept}")

out = {nm: {"ic1": results[nm]["ic1"], "ic5": results[nm]["ic5"], "ic10": results[nm]["ic10"],
            "cov": results[nm]["cov"], "to": results[nm]["to"],
            "max_lib_rho": results[nm].get("max_lib_rho"), "max_lib_rho_id": results[nm].get("max_lib_rho_id")}
       for nm in kept}
json.dump(out, open("scripts/miner2_screen_cycle4_results.json", "w"), indent=1, default=str)
print(f"saved cycle4 results for {len(kept)} kept candidates | {time.time()-T0:.1f}s")
