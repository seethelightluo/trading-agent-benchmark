"""miner_3 batch AB (2032-04-05) - deep validation of batch AA full-pass candidates.

Visible data through 2032-04-02. Universe: 15 tradable cross-asset instruments.
Gates: |IC|>=0.0070 and |ICIR|>=0.0840 at h=10; min IC dates >= 120 for admission.
Reports decay profile, library max-abs correlation, regime/yearly splits.
No live-account interaction.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)

hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
op = pd.concat({a: panels[a]["open"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)

H = 10
fwd = forward_returns(closes, H)
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()


def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)


def rolling_corr_series(a, b, win=20, min_obs=15):
    """a: DataFrame, b: Series -> DataFrame of rolling corr per column."""
    out = {}
    for col in a.columns:
        z = pd.concat([a[col].rename("a"), b.rename("b")], axis=1).dropna()
        c = z["a"].rolling(win).corr(z["b"])
        out[col] = c.where(z["b"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=a.index)


# ---------------- candidate signals (batch AA full-pass) ----------------
C = {}
C["updown_vol_ratio_20"] = rets.clip(lower=0).rolling(20).std() / (rets.clip(upper=0).rolling(20).std() + 1e-12)
C["trend_tstat_60"] = (closes/closes.shift(60)-1) / (vol60 * np.sqrt(60) + 1e-12)
crypto_basket = rets[["BTC", "ETH"]].mean(axis=1)
C["crypto_beta_60"] = rolling_beta(rets, crypto_basket, 60)
C["max_gain_20"] = rets.rolling(20).max()

# reference library signals (effective + historical/evicted) for correlation audit
lib = {}
lib["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["vol_ratio_20_60"] = vol20 / vol60
lib["volume_z_20"] = (vol_panel - vol_panel.rolling(60).mean()) / (vol_panel.rolling(60).std() + 1e-12)
lib["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
lib["vix_beta_cond_60x20"] = rolling_beta(rets, vix.pct_change(), 60) * (vix.pct_change().rolling(20).mean() > 0).astype(float)
lib["mom_vs_median_60d"] = (closes/closes.shift(60)-1) - (closes/closes.shift(60)-1).rolling(60).median()
lib["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60) * (closes["US10Y"].pct_change().rolling(60).mean() > 0).astype(float)
lib["downside_ratio_60d"] = rets.clip(upper=0).rolling(60).std() / (rets.rolling(60).std() + 1e-12)
lib["max_dd_60d"] = closes / closes.rolling(60).max() - 1.0
lib["kurt_60d"] = rets.rolling(60).kurt()
lib["corr_asset_mkt_20"] = rolling_corr_series(rets, mkt_ret, 20)
lib["vix_corr_20"] = rolling_corr_series(rets, vix.pct_change(), 20)
lib["hl_pos_20d"] = (hi - lo) / (closes + 1e-12)
lib["rsi_14"] = closes.rolling(14).apply(lambda s: (s.diff().clip(lower=0).sum() / (s.diff().abs().sum() + 1e-12)), raw=False)
lib["xau_beta_60"] = rolling_beta(rets, rets["XAU"], 60)
lib["comm_beta_60"] = rolling_beta(rets, rets[["WTI", "XAU", "COPPER"]].mean(axis=1), 60)
lib["spread_beta_cnus_60"] = rolling_beta(rets, (closes["CN10Y"] - closes["US10Y"]).diff(), 60)

print(f"library {len(lib)} signals, candidates {len(C)}; {time.time()-t0:.1f}s", flush=True)


def pairwise_max_abs_corr(cand, libsig):
    """max |spearman rho| over assets of cand vs a library signal."""
    common = cand.columns.intersection(libsig.columns)
    if len(common) == 0:
        return np.nan
    rhos = []
    for a in common:
        pair = pd.concat([cand[a].rename("c"), libsig[a].rename("l")], axis=1).dropna()
        if len(pair) > 60 and pair["c"].std() > 1e-14 and pair["l"].std() > 1e-14:
            rhos.append(pair["c"].corr(pair["l"], method="spearman"))
    if not rhos:
        return np.nan
    return float(np.max(np.abs(rhos)))


def full_report(name, sig, expected_sign=1):
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=expected_sign)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    dec = decay_profile(sig, closes, horizons=[1, 2, 3, 5, 10, 20])
    mc = {k: pairwise_max_abs_corr(sig, v) for k, v in lib.items()}
    mc = {k: v for k, v in mc.items() if v == v}
    mcl = max(mc.values()) if mc else np.nan
    mcl_name = max(mc, key=mc.get) if mc else ""
    # yearly / regime splits
    yr = {}
    for y in range(2020, 2033):
        sub = ics[(ics.index.year == y)]
        if len(sub) > 20:
            mm, ss = sub.mean(), sub.std(ddof=1)
            yr[y] = (round(mm, 4), round(mm/ss, 2) if ss and ss > 0 else np.nan)
    print(f"\n=== {name} (expected_sign={expected_sign:+d}) ===", flush=True)
    print(f"  IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"std={s['ic_std']:.3f} | cov_dates={cov['coverage_dates_ge8']:.2f} cov_assets={cov['coverage_asset_days']:.2f} to10={to if to is not None else float('nan'):.2f}", flush=True)
    print(f"  decay(1,2,3,5,10,20)={[round(x,4) if x==x else None for x in dec]}", flush=True)
    print(f"  max_abs_lib_corr={mcl:.3f} (vs {mcl_name})", flush=True)
    print(f"  yearly_ic(mean,icir): {yr}", flush=True)
    return s, ics, sig, mc


results = {}
for name, sig in C.items():
    es = 1
    s, ics, sig, mc = full_report(name, sig, expected_sign=es)
    results[name] = (s, ics, sig, mc)

# deep re-validation of current effective factors with regime split
print("\n\n########## RE-VALIDATION DEEP DIVE (current effective) ##########", flush=True)
existing = {
    "vol_adj_mom_accel_20x60": ((closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std(), 1),
    "dn_mkt_beta_60d": (rolling_beta(rets, mkt_ret.clip(upper=0), 60), 1),
    "rate_beta_cn10y_60d": (rolling_beta(rets, closes["CN10Y"].pct_change(), 60), -1),
}
for name, (sig, es) in existing.items():
    s, ics, _, _ = full_report(name, sig, expected_sign=es)
    # recent direction check: raw IC over last 63/126 days vs full
    raw_mean = ics.mean()
    r63 = ics.iloc[-63:].mean()
    r126 = ics.iloc[-126:].mean()
    print(f"  >> {name}: raw full IC={raw_mean:+.4f} vs r126={r126:+.4f} vs r63={r63:+.4f} "
          f"(sign consistency {'OK' if np.sign(raw_mean)==np.sign(r63) else 'FLIPPED'})", flush=True)

print(f"\ntotal time {time.time()-t0:.1f}s", flush=True)
