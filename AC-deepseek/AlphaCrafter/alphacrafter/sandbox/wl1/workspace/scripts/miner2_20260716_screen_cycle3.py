"""
miner_2 cycle 3 (2026-07-16): screen NOVEL factor families for library diversity.
Families: volume/liquidity, volatility structure, trend/drawdown, overnight-intraday
decomposition, and cross-asset beta-timing. Persist passers with .npy artifacts.
"""
import sys, os, json, pickle, time, base64, gzip
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
dxy = macro["DXY"].reindex(idx).ffill()
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

# ---------------- library signals aligned to master index ----------------
def load_library_signals():
    signals = {}
    for p in sorted(__import__("glob").glob("factors/*.json")):
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

# ---------------- candidate panels ----------------
P = {}
# --- volume / liquidity family ---
zv = (lvol - lvol.rolling(60).mean()) / lvol.rolling(60).std()
P["vol_z_60"] = zv
P["vol_z_20"] = (lvol - lvol.rolling(20).mean()) / lvol.rolling(20).std()
P["vol_trend_20_60"] = lvol.rolling(20).mean() / lvol.rolling(60).mean() - 1
amihud = (ret.abs() / vol.clip(lower=1e-9))
amihud_z = (amihud - amihud.rolling(60).mean()) / amihud.rolling(60).std()
P["neg_amihud_z_60"] = -amihud_z
dvol = np.log(vol).diff()
vc = ret.rolling(20).corr(dvol)
P["ret_vol_corr_20"] = vc

# --- volatility structure family ---
P["skew_20"] = ret.rolling(20).skew()
P["down_vol_ratio_20"] = (ret.clip(upper=0).rolling(20).std() / ret.rolling(20).std()) - 1
vol20 = ret.rolling(20).std()
P["vol_of_vol_20x60"] = vol20.rolling(60).std() / vol20.rolling(60).mean() - 1
P["vol_ratio_20_60"] = vol20 / ret.rolling(60).std() - 1
P["neg_vol_persist_20"] = -(ret.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False))

# --- trend / structure family ---
P["dist_high_252"] = close / close.rolling(252).max() - 1
P["drawdown_60"] = close / close.rolling(60).max() - 1
P["ma_dist_20"] = close / close.rolling(20).mean() - 1
P["mom_12m_1m"] = np.log(close / close.shift(252)) - np.log(close / close.shift(21))
P["mom_60d_skip5"] = np.log(close / close.shift(60)) - np.log(close / close.shift(5))
P["neg_dist_high_252"] = -(close / close.rolling(252).max() - 1)

# --- overnight / intraday decomposition ---
on_ret = np.log(open_ / close.shift(1))
id_ret = np.log(close / open_)
P["on_ret_5"] = on_ret.rolling(5).mean()
P["id_ret_5"] = id_ret.rolling(5).mean()
P["on_id_vol_ratio_20"] = on_ret.rolling(20).std() / id_ret.rolling(20).std() - 1
P["gap_rev_1d"] = -on_ret
P["body_rev_1d"] = -id_ret

# --- cross-asset beta timing ---
ew = ret.mean(axis=1)
beta_panel_60 = ret.rolling(60).cov(ew) / ew.rolling(60).var()
ew_mom20 = ew.rolling(20).sum()
P["beta_timing_60x20"] = beta_panel_60 * ew_mom20
vix_beta_60 = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
P["neg_vix_beta_cond"] = -vix_beta_60 * (vix / vix.rolling(252).mean() - 1)

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

# library correlation for passers
for nm in passers:
    rho, fid = max_lib_rho(results[nm]["panel"].loc[m])
    results[nm]["max_lib_rho"] = rho; results[nm]["max_lib_rho_id"] = fid
    print(f"  {nm:22s} max_lib_rho={rho:.3f} vs {fid}")

# greedy diverse selection
def quality(nm):
    r = results[nm]
    return abs(r["ic1"]["ic"]) * abs(r["ic1"]["icir"])

kept = []
for nm in sorted(passers, key=lambda x: -quality(x)):
    r = results[nm]
    if r["max_lib_rho"] >= RHO_MAX:
        print(f"  drop {nm}: too close to library ({r['max_lib_rho']:.3f})")
        continue
    if all(pair_rho(results[nm]["panel"].loc[m], results[k]["panel"].loc[m]) < RHO_MAX for k in kept):
        kept.append(nm)
print(f"\ndiverse kept: {kept}")

# save screening results for persistence step
out = {nm: {"ic1": results[nm]["ic1"], "ic5": results[nm]["ic5"], "ic10": results[nm]["ic10"],
            "cov": results[nm]["cov"], "to": results[nm]["to"],
            "max_lib_rho": results[nm].get("max_lib_rho"), "max_lib_rho_id": results[nm].get("max_lib_rho_id")}
       for nm in kept}
json.dump(out, open("scripts/miner2_screen_cycle3_results.json", "w"), indent=1, default=str)
print(f"saved cycle3 results for {len(kept)} kept candidates | {time.time()-T0:.1f}s")
