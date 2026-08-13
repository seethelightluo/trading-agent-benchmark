"""miner_2 batch Z finish v2 (2031-09-08) - optimized full validation package.

Completes batch Z screening (ETH-BTC relative-momentum candidate dropped: it is a
single time series, not a cross-sectional signal, so rank IC is undefined by
construction) and computes the full validation package for ALL IC/ICIR passers
from batch Y + Z: pairwise spearman rho among passers, rho vs the 3 live
factors, and max_abs_library_correlation vs the full historical library.

Admission gates (h=10, min_valid=8): |IC| >= 0.0070 and |ICIR| >= 0.0840.
Correlation conflict threshold 0.5 per worldline pairwise signal-quality contract.
Visible data only through the previous completed trading day. No live-account
interaction.
"""
import sys, time, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, TRADABLE)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | load {time.time()-t0:.1f}s", flush=True)


def align(series, idx):
    return series.reindex(idx).ffill()


vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)

H = 10


def fast_rank_ic(factor_panel, fwd, min_valid=8):
    """Vectorized daily Spearman rank IC (cross-section per date)."""
    idx = factor_panel.index.intersection(fwd.index)
    f = factor_panel.loc[idx].rank(axis=1, method="average")
    r = fwd.loc[idx].rank(axis=1, method="average")
    mm = (factor_panel.loc[idx].notna() & fwd.loc[idx].notna()).to_numpy(dtype=bool)
    n = mm.sum(axis=1).astype(float)
    fm = np.where(mm, f.to_numpy(dtype=float), np.nan)
    rm = np.where(mm, r.to_numpy(dtype=float), np.nan)
    fmean = np.nanmean(fm, axis=1)
    rmean = np.nanmean(rm, axis=1)
    fstd = np.nanstd(fm, axis=1, ddof=1)
    rstd = np.nanstd(rm, axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.nansum((fm - fmean[:, None]) * (rm - rmean[:, None]), axis=1) / (n - 1)
        ic = cov / (fstd * rstd)
    keep = (n >= min_valid) & np.isfinite(ic) & (fstd > 1e-14) & (rstd > 1e-14)
    return pd.Series(ic[keep], index=idx[keep], name="ic")


def summarize_ic(ics, expected_sign=1):
    if len(ics) == 0:
        return {"ic": 0.0, "icir": 0.0, "ic_hit_ratio": 0.0, "n_ic_dates": 0, "ic_std": 0.0}
    s = ics.std(ddof=1)
    return {
        "ic": round(float(ics.mean()), 4),
        "icir": round(float(ics.mean() / s), 4) if s > 0 else 0.0,
        "ic_hit_ratio": round(float((np.sign(ics) == expected_sign).mean()), 3),
        "n_ic_dates": int(len(ics)),
        "ic_std": round(float(s), 4),
    }


def decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8, expected_sign=1):
    out = {}
    for h in horizons:
        ics = fast_rank_ic(sig, forward_returns(closes, h), min_valid)
        out[str(h)] = round(float(ics.mean()), 4) if len(ics) else None
    return out


def recent_ic(ics, w):
    sub = ics.iloc[-w:]
    if len(sub) < 3:
        return (None, None)
    s = sub.std(ddof=1)
    return (round(float(sub.mean()), 4), round(float(sub.mean() / s), 3) if s > 0 else None)


def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        out[a] = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=y.index)


def pooled_spearman(a, b):
    both = pd.concat([a.stack().rename("a"), b.stack().rename("b")], axis=1).dropna()
    if len(both) < 30:
        return float("nan")
    return float(both["a"].rank().corr(both["b"].rank()))


# ---------------- 0) RE-VALIDATE current effective factors ----------------
print("\n=== RE-VALIDATION of current effective factors (h=10) ===", flush=True)
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
vol120 = rets.rolling(120).std()

live = {}
live["vol_adj_mom_accel_20x60"] = (closes / closes.shift(20) - 1 - (closes / closes.shift(60) - 1)) / vol20
live["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
live["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)

fwd10 = forward_returns(closes, H)
for name, sig in live.items():
    exp = 1 if name != "rate_beta_cn10y_60d" else -1
    ics = fast_rank_ic(sig, fwd10)
    s = summarize_ic(ics, exp)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    r63 = recent_ic(ics, 63)
    r252 = recent_ic(ics, 252)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({r63[0]},{r63[1]}) r252=({r252[0]},{r252[1]}) cov={cov['coverage_dates_ge8']:.2f} to={to}{flag}", flush=True)

# ---------------- 1) CANDIDATE EVAL (batch Y passers + batch Z) ----------------
print("\n=== FULL CANDIDATE EVAL (h=10) ===", flush=True)
C = {}


def downside_ratio(r, win=60):
    out = {}
    for a in r.columns:
        v = r[a]
        out[a] = v.clip(upper=0).rolling(win).std() / (v.rolling(win).std() + 1e-12)
    return pd.DataFrame(out, index=r.index)


C["downside_ratio_60d"] = downside_ratio(rets, 60)
C["beta_btc_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)
C["drawdown_60d"] = (closes - closes.rolling(60).max()) / closes.rolling(60).max()
C["hl_pos_20d"] = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min())
vix_ret = vix.pct_change()
C["vix_beta_x_level_60d"] = -rolling_beta(rets, vix_ret, 60) * (vix / vix.shift(60)).to_frame(0).values
C["mom60_skip5_voladj"] = (closes.shift(5) / closes.shift(65) - 1) / vol60
C["sharpe_120d"] = (closes / closes.shift(120) - 1) / (vol120 + 1e-12)
C["drawdown_120d"] = (closes - closes.rolling(120).max()) / closes.rolling(120).max()
C["rev5_voladj"] = -(closes / closes.shift(5) - 1) / vol20
C["vol_ts_slope"] = (vol20 - vol60) / (vol60 + 1e-12)
C["skew_60d"] = rets.rolling(60).skew()
C["xau_dn_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change() * (mkt_ret < 0).astype(float), 60)
C["beta_usdcny_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)

DIR = {"drawdown_60d": -1, "hl_pos_20d": -1, "mom60_skip5_voladj": -1,
       "sharpe_120d": -1, "drawdown_120d": -1, "vol_ts_slope": -1}

results = {}
for name, sig in C.items():
    exp = DIR.get(name, 1)
    ics = fast_rank_ic(sig, fwd10)
    s = summarize_ic(ics, exp)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), expected_sign=exp)
    s.update(cov)
    s["turnover_10d_rank"] = to
    s["decay_ic_by_horizon"] = dec
    results[name] = (s, ics, sig)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to} dec10={dec['10']}{flag}", flush=True)

passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"\nFull-pass count: {len(passing)}", flush=True)

# ---------------- 2) PAIRWISE RHO AMONG PASSERS ----------------
names = list(passing.keys())
if names:
    print("\n=== PAIRWISE SPEARMAN RHO (passers) ===", flush=True)
    rho_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            rho_mat.loc[i, j] = round(pooled_spearman(passing[i][2], passing[j][2]), 4) if i != j else 1.0
    print(rho_mat.round(3).to_string(), flush=True)

    print("\n=== RHO vs LIVE LIBRARY (3 effective) ===", flush=True)
    for name in names:
        row = {ln: round(pooled_spearman(passing[name][2], lsig), 4) for ln, lsig in live.items()}
        mx = max(row.values(), key=abs) if row else 0.0
        print(f"{name:24s} {row}  max_abs={mx:.4f}", flush=True)

    # ---------------- 3) MAX ABS LIBRARY CORRELATION (full historical lib) ----------------
    lib_full = dict(live)
    lib_full["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
    lib_full["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
    lib_full["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    vix_beta = rolling_beta(rets, vix_ret, 60)
    lib_full["vix_beta_cond_60x20"] = -vix_beta * (vix / vix.shift(20) - 1.0)
    lib_full["vol_price_corr_20"] = rets.rolling(20).corr(mkt_ret)
    lib_full["vol_ratio_20_60"] = vol20 / vol60
    lib_full["rsi_14"] = 100 - 100 / (1 + (rets.clip(lower=0).rolling(14).mean()) /
                                      ((-rets.clip(upper=0)).rolling(14).mean() + 1e-9))
    lib_full["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60)
    lib_full["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
    lib_full["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
    vol_df = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index()
    lib_full["volume_z_20"] = (vol_df - vol_df.rolling(60).mean()) / (vol_df.rolling(60).std() + 1e-12)

    print("\n=== MAX_ABS_LIBRARY_CORRELATION (full historical lib incl. evicted) ===", flush=True)
    for name in names:
        best, key = 0.0, None
        for ln, lsig in lib_full.items():
            if ln == name:
                continue
            r = pooled_spearman(passing[name][2], lsig)
            if not np.isnan(r) and abs(r) > best:
                best, key = abs(r), ln
        passing[name][0]["max_abs_library_correlation"] = round(best, 4)
        passing[name][0]["max_corr_factor"] = key
        print(f"{name:24s} max_abs_lib_corr={best:.4f} (vs {key})", flush=True)

# ---------------- SAVE RESULTS ----------------
out = {k: {kk: vv for kk, vv in v[0].items()} for k, v in results.items()}
out["_meta"] = {"asof": str(closes.index.max().date()), "n_assets": closes.shape[1],
                "gates": {"abs_ic": 0.0070, "abs_icir": 0.0840, "min_valid": 8, "h": H}}
with open("scripts/_miner2_20310908_batchZ_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nsaved scripts/_miner2_20310908_batchZ_results.json | done {time.time()-t0:.1f}s", flush=True)
