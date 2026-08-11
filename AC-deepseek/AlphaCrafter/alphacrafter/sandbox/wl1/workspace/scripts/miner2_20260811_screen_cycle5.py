"""
miner_2 cycle 5 (2026-08-11): re-screen NOVEL factor families with corrected construction.
Fixes vs cycle4:
  - union-calendar panel has ~1/3 NaN per symbol -> rolling() needs explicit min_periods
  - DataFrame*Series broadcast bug -> .mul(series, axis=0)
  - volume columns only populated for BTC/ETH -> volume factors dropped (reported)
Families: extremes/lottery, dist-from-low, volatility compression, gap structure,
momentum term structure, return asymmetry (mean/vol/skew/kurt), autocorrelation,
Hurst persistence, dispersion-conditional reversal, VIX-regime reversal, range position, ATR level.
Gate: |IC1| >= 0.0070 and |ICIR1| >= 0.0840; rho < 0.5 vs library and pairwise.
Persists passers immediately to factors/<fid>.json with full schema + signal artifact.
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
NOW = "2026-08-11"

cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close = cache["close"][SYMBOLS]; open_ = cache["open"][SYMBOLS]
high = cache["high"][SYMBOLS]; low = cache["low"][SYMBOLS]
vol = cache["vol"][SYMBOLS]
idx = close.index
ret = np.log(close).diff()
macro = cache["macro"]
vix = macro["VIX"].reindex(idx).ffill()

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
    for p in glob.glob("factors/*.json"):
        if "_deprecated" in p or ".bak" in p:
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

# ---------------- candidate panels (novel families, fixed construction) ----------------
P = {}
r1 = ret.shift(1)

def ro(df, w, mp=None):
    """rolling with sane min_periods for union calendar."""
    return df.rolling(w, min_periods=(mp if mp is not None else max(3, w // 3)))

# 1. extremes / lottery family
P["neg_max_ret_20"] = -ro(ret, 20).max()
P["min_ret_20"] = ro(ret, 20).min()
P["extreme_spread_20"] = ro(ret, 20).max() - ro(ret, 20).min()

# 2. distance from low
P["dist_low_252"] = close / ro(close, 252, 120).min() - 1
P["dist_low_60"] = close / ro(close, 60, 20).min() - 1

# 3. volatility compression
mid20 = ro(close, 20).mean(); sd20 = ro(close, 20).std()
P["bb_bandwidth_20"] = 4 * sd20 / mid20
rng = high - low
P["nr7_neg"] = -(rng / ro(rng, 7, 4).min())

# 4. gap structure
on_ret = np.log(open_ / close.shift(1))
id_ret = np.log(close / open_)
P["gap_follow_1d"] = on_ret * id_ret
gap_z20 = (on_ret - ro(on_ret, 20).mean()) / ro(on_ret, 20).std()
P["gap_z_rev_20"] = -gap_z20

# 5. momentum term structure
P["mom_term_5_20"] = np.log(close / close.shift(5)) - np.log(close / close.shift(20))

# 6. return asymmetry
up = ret.clip(lower=0); dn = ret.clip(upper=0)
P["asym_mean_20"] = ro(up, 20).mean() + ro(dn, 20).mean()
P["up_vol_ratio_20"] = ro(ret.where(ret > 0), 20).std() / ro(ret.where(ret < 0), 20).std() - 1
P["skew_20"] = ret.rolling(20, min_periods=8).skew()
P["kurt_20"] = ret.rolling(20, min_periods=8).kurt()

# 7. autocorrelation / persistence
P["autocorr_5"] = ret.rolling(5, min_periods=3).apply(
    lambda x: pd.Series(x).autocorr(lag=1) if np.isfinite(x).sum() >= 3 else np.nan, raw=True)

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

P["hurst_20"] = ret.rolling(20, min_periods=10).apply(lambda x: hurst_est(np.asarray(x)), raw=False)

# 8. dispersion-conditional reversal (fixed broadcast)
disp = ret.std(axis=1)
disp_z = (disp - disp.rolling(60, min_periods=20).mean()) / disp.rolling(60, min_periods=20).std()
P["disp_cond_rev_1d"] = r1.mul(-disp_z.shift(1), axis=0)

# 9. VIX-regime reversal (fixed broadcast)
vix_z = (vix / vix.rolling(252, min_periods=120).mean() - 1).shift(1)
P["vix_reg_rev_1d"] = r1.mul(-(1 + 2 * vix_z.clip(lower=0)), axis=0)

# 10. 20d range position (longer-horizon close-location)
rng20_hi = ro(high, 20).max(); rng20_lo = ro(low, 20).min()
P["range_pos_20"] = (close - rng20_lo) / (rng20_hi - rng20_lo) - 0.5

# 11. ATR level
tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
P["atr_ratio_20"] = ro(tr, 20).mean() / ro(close, 20).mean()

# 12. growth-signal beta (COPPER/XAU risk-on)
gx = np.log(close["COPPER"] / close["XAU"]).diff()
gx_mom20 = np.log(close["COPPER"] / close["XAU"]).diff(20)
gx_prev = gx.shift(1)
beta_gx60 = ret.rolling(60, min_periods=20).cov(gx_prev) / gx_prev.rolling(60, min_periods=20).var()
P["growth_beta_60x20"] = beta_gx60 * gx_mom20.shift(1)

print(f"candidates: {len(P)}")

results = {}
for nm, p in P.items():
    Pp = p.loc[m]
    ic1 = fast_ic(Pp, fwd1.loc[m]); ic5 = fast_ic(Pp, fwd5.loc[m]); ic10 = fast_ic(Pp, fwd10.loc[m])
    cov = float(Pp.notna().sum().sum()) / n_cells
    to = turnover10(Pp)
    passed = (abs(ic1["ic"]) >= IC_MIN) and (abs(ic1["icir"]) >= ICIR_MIN) and (ic1["n_dates"] > 200)
    results[nm] = {"panel": Pp, "ic1": ic1, "ic5": ic5, "ic10": ic10, "cov": cov, "to": to, "passed": passed}
    print(f"{nm:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit1={ic1['hit']:.3f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} | IC10={ic10['ic']:+.4f} | "
          f"{'PASS' if passed else 'fail'}")

passers = [nm for nm, r in results.items() if r["passed"]]
print(f"\npassing gate: {passers}")

for nm in passers:
    rho, fid = max_lib_rho(results[nm]["panel"])
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
    if all(pair_rho(results[nm]["panel"], results[k]["panel"]) < RHO_MAX for k in kept):
        kept.append(nm)
print(f"\ndiverse kept: {kept}")

# ---------------- persistence of passers (mandatory) ----------------
def make_artifact(panel):
    pp = panel.copy()
    data = pp.values.astype(np.float32)
    comp = gzip.compress(data.tobytes())
    return {
        "format": "gzip+float32", "symbols": SYMBOLS, "n_dates": int(data.shape[0]),
        "n_symbols": int(data.shape[1]), "date_start": str(pp.index[0].date()),
        "date_end": str(pp.index[-1].date()), "data_b64": base64.b64encode(comp).decode(),
        "recovery": "reshape(n_dates,n_symbols) on union calendar; NaN = non-trading day"
    }

def by_year_ic1(panel):
    out = {}
    for y in range(2021, 2027):
        sub = panel.loc[panel.index.year == y]
        if len(sub) < 50:
            continue
        r = fast_ic(sub, fwd1.loc[sub.index])
        if r["n_dates"] > 0:
            out[str(y)] = {"ic": r["ic"], "icir": r["icir"], "n_dates": r["n_dates"]}
    return out

for nm in kept:
    r = results[nm]
    Pp = r["panel"]
    fid = f"miner2_20260811_{nm}"
    factor_doc = {
        "factor_id": fid,
        "factor_name": nm.replace("_", " ").title(),
        "version": "1.0",
        "calculation": {
            "expression": f"see script miner2_20260811_screen_cycle5.py; family={nm}",
            "description": f"Novel cross-asset factor '{nm}' mined on the 15-name benchmark "
                           "universe; rolling windows use min_periods adapted to the union calendar."
        },
        "dependencies": ["close", "open", "high", "low"],
        "parameters": {"min_periods_rule": "max(3, window//3); 252d->120, 60d->20, 7d->4",
                       "validation_window": "2021-01-01..2026-07-15"},
        "validation": {
            "status": "EFFECTIVE",
            "admission_gate": {"ic_min": IC_MIN, "icir_min": ICIR_MIN,
                               "rho_max_library": RHO_MAX, "min_names_per_date": MIN_NAMES},
            "period": {"start": str(VALID_LO.date()), "end": str(VALID_HI.date())},
            "last_validated": NOW,
            "metrics": {
                "ic1": r["ic1"]["ic"], "icir1": r["ic1"]["icir"], "hit1": r["ic1"]["hit"],
                "n_dates": r["ic1"]["n_dates"], "n_obs": r["ic1"]["n_obs"],
                "ic5": r["ic5"]["ic"], "icir5": r["ic5"]["icir"],
                "ic10": r["ic10"]["ic"], "icir10": r["ic10"]["icir"],
                "coverage": r["cov"], "turnover_10d": r["to"],
                "max_abs_library_correlation": r["max_lib_rho"],
                "max_abs_library_correlation_id": r["max_lib_rho_id"]
            },
            "by_year_ic1": by_year_ic1(Pp),
            "regime_notes": "2021-2026 span: post-COVID recovery, 2022 rate shock, 2023-24 AI rally, "
                            "2025-26 crypto/commodity cycles; cross-asset union calendar with ~1/3 NaN per symbol.",
            "timeliness": {"validated_on": NOW, "decay_observed": None}
        },
        "tags": [nm.split("_")[0], "cross-asset", "novel"],
        "provenance": {"miner": "miner_2", "cycle": 5, "script": "scripts/miner2_20260811_screen_cycle5.py"},
        "signal_artifact": make_artifact(Pp),
        "benchmark_admission": {
            "gate_version": "ac-worldline-v2-migration-gate",
            "ic1": r["ic1"]["ic"], "icir1": r["ic1"]["icir"],
            "ic1_abs_ok": abs(r["ic1"]["ic"]) >= IC_MIN,
            "icir1_abs_ok": abs(r["ic1"]["icir"]) >= ICIR_MIN,
            "status": "PASS"
        }
    }
    path = f"factors/{fid}.json"
    json.dump(factor_doc, open(path, "w"))
    print(f"PERSISTED {path} ({os.path.getsize(path)} bytes)")

out = {nm: {"ic1": results[nm]["ic1"], "ic5": results[nm]["ic5"], "ic10": results[nm]["ic10"],
            "cov": results[nm]["cov"], "to": results[nm]["to"],
            "max_lib_rho": results[nm].get("max_lib_rho"), "max_lib_rho_id": results[nm].get("max_lib_rho_id")}
       for nm in kept}
json.dump(out, open("scripts/miner2_screen_cycle5_results.json", "w"), indent=1, default=str)
print(f"saved cycle5 results for {len(kept)} kept candidates | {time.time()-T0:.1f}s")
