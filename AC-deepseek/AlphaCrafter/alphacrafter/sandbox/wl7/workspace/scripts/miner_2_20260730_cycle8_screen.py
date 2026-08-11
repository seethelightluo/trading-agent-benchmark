"""miner_2 cycle 8 screening: re-validate prior passers vs FULL 11-factor library + novel families.

Context: cycle2/3 identified usdcny_beta_60, dxy_beta_60, vol_price_corr_20, body_ratio_20,
sharpe_60 as IC/ICIR passers but they were never persisted / audited against the current
11-factor library. This round re-validates them AND tests novel families far from library:
  - usdcny_beta_cond_60x20 : beta(asset,USDCNY,60) * USDCNY 20d return (CNY FX conditional)
  - dxy_beta_cond_60x20    : beta(asset,DXY,60) * DXY 20d return
  - vol_price_corr_20      : 20d corr(volume, pct_change) (volume-price confirmation)
  - body_ratio_20          : 20d mean |close-open| / (high-low) (candle shape)
  - sharpe_60              : 60d mean(ret)/std(ret) (risk-adjusted return)
  - gap_freq_20            : 20d fraction of |open/prev_close-1| > 1.5% (jump frequency)
  - tail_ratio_20          : 95th pct |ret| / median |ret| over 20d (heavy-tail)
  - drawup_60              : close/rolling_min(close,60)-1 (upside distance)
  - downside_beta_ratio_60 : beta_down/beta_up vs EW index (asymmetric risk)
  - overnight_mom_20       : 20d compounded overnight return (open/prev_close-1)
Admission gate h=10: |IC|>=0.007, |ICIR|>=0.084; library corr < 0.5 (both stacked and per-date).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
import miner_2_lib as lib

WATCH = lib.WATCH
MAX_VISIBLE = lib.MAX_VISIBLE
FACTOR_LAST = lib.FACTOR_LAST
MIN_ASSETS = lib.MIN_ASSETS
ADMISSION = lib.ADMISSION
EPS = 1e-12

panel = lib.load_panel()
rets = panel.pct_change()
mac = lib.load_macro()


def load_ohlcv_assets():
    """Per-asset OHLCV frames (own calendar, visible data only)."""
    out = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[s] = df.astype(float)
    return out


ohlcv = load_ohlcv_assets()


def per_asset_series(fn):
    out = {}
    for s in WATCH:
        c = panel[s].dropna()
        out[s] = fn(c)
    return pd.DataFrame(out, index=panel.index)


def per_asset_ohlcv(fn):
    out = {}
    for s in WATCH:
        df = ohlcv[s]
        out[s] = fn(df)
    return pd.DataFrame(out, index=panel.index)


# ---------------- library (full 11) ----------------
def library_signals_full():
    libs = {}
    r = rets
    libs["amihud_20"] = (r.abs() / (ohlcv_vol_panel() + EPS)).rolling(20).mean()
    ew = panel.mean(axis=1)
    ew_r = ew.pct_change()
    cols = {}
    for a in WATCH:
        s = panel[a].dropna()
        er = ew_r.reindex(s.index)
        z = pd.concat([s.pct_change().rename("r"), er.rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    libs["beta_ew_60d"] = pd.DataFrame(cols, index=panel.index)

    def dsvr(s):
        rr = s.pct_change()
        down = rr.where(rr < 0, 0.0)
        ds = np.sqrt((down ** 2).rolling(20).mean())
        tot = rr.rolling(20).std()
        return -(ds / tot)
    libs["downside_vol_ratio_20"] = per_asset_series(dsvr)
    libs["max_ret_20d"] = r.rolling(20).max()
    libs["mom_10d_skip5"] = per_asset_series(lambda s: s.shift(5) / s.shift(15) - 1.0)
    libs["mom_120d_skip5"] = per_asset_series(lambda s: s.shift(5) / s.shift(125) - 1.0)
    m20 = per_asset_series(lambda s: s.shift(5) / s.shift(25) - 1.0)
    libs["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    vix = mac["VIX"].dropna()
    vix20 = vix / vix.shift(20) - 1.0
    cols = {}
    for a in WATCH:
        s = panel[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), vix.pct_change().reindex(s.index).rename("v")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
        cols[a] = (-beta * vix20.reindex(s.index))
    libs["vix_beta_cond_60x20"] = pd.DataFrame(cols, index=panel.index)
    libs["vol_adj_mom_20x60"] = per_asset_series(
        lambda s: (s.shift(5) / s.shift(25) - 1.0) / s.pct_change().rolling(60).std())
    libs["vol_of_vol20x60"] = r.rolling(20).std().rolling(60).std()
    eur = mac["EURUSD"].dropna()
    eur20 = eur / eur.shift(20) - 1.0
    cols = {}
    for a in WATCH:
        s = panel[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), eur.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
        cols[a] = (beta * eur20.reindex(s.index))
    libs["eurusd_beta_cond_60x20"] = pd.DataFrame(cols, index=panel.index)
    return libs


def ohlcv_vol_panel():
    cols = {}
    for s in WATCH:
        cols[s] = ohlcv[s]["volume"]
    return pd.concat(cols, axis=1, sort=True).reindex(panel.index)


def macro_beta_cond(mname):
    m = mac[mname].dropna()
    mr = m.pct_change()
    m_ret20 = m / m.shift(20) - 1.0
    cols = {}
    for a in WATCH:
        s = panel[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), mr.reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var().replace(0, np.nan)
        cols[a] = beta * m_ret20.reindex(s.index)
    return pd.DataFrame(cols, index=panel.index)


# ---------------- candidates ----------------
def cand_usdcny_beta_cond(panel_):
    return macro_beta_cond("USDCNY")


def cand_dxy_beta_cond(panel_):
    return macro_beta_cond("DXY")


def cand_vol_price_corr_20(panel_):
    def f(df):
        r = df["close"].pct_change()
        return r.rolling(20).corr(df["volume"])
    return per_asset_ohlcv(f)


def cand_body_ratio_20(panel_):
    def f(df):
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        body = (df["close"] - df["open"]).abs() / (rng + EPS)
        return body.rolling(20, min_periods=10).mean()
    return per_asset_ohlcv(f)


def cand_sharpe_60(panel_):
    def f(s):
        rr = s.pct_change()
        return rr.rolling(60).mean() / rr.rolling(60).std().replace(0, np.nan)
    return per_asset_series(f)


def cand_gap_freq_20(panel_):
    def f(df):
        gap = (df["open"] / df["close"].shift(1) - 1.0).abs()
        return (gap > 0.015).astype(float).rolling(20, min_periods=10).mean()
    return per_asset_ohlcv(f)


def cand_tail_ratio_20(panel_):
    def f(s):
        rr = s.pct_change().abs()
        q95 = rr.rolling(20, min_periods=10).quantile(0.95)
        med = rr.rolling(20, min_periods=10).median()
        return q95 / med.replace(0, np.nan)
    return per_asset_series(f)


def cand_drawup_60(panel_):
    return per_asset_series(lambda s: s / s.rolling(60, min_periods=10).min() - 1.0)


def cand_downside_beta_ratio_60(panel_):
    ew = panel.mean(axis=1)
    ew_r = ew.pct_change()
    out = {}
    for a in WATCH:
        s = panel[a].dropna()
        r = s.pct_change()
        z = pd.concat([r.rename("r"), ew_r.reindex(s.index).rename("m")], axis=1).dropna()
        z["dm"] = (z["m"] < 0).astype(float)
        z["um"] = (z["m"] > 0).astype(float)
        # rolling beta on down days / up days (beta denominator: m var on same days)
        bd = (z["r"] * z["dm"]).rolling(60).sum() / (z["m"] * z["dm"]).rolling(60).sum().replace(0, np.nan)
        bu = (z["r"] * z["um"]).rolling(60).sum() / (z["m"] * z["um"]).rolling(60).sum().replace(0, np.nan)
        out[a] = bd / bu.replace(0, np.nan)
    return pd.DataFrame(out, index=panel.index)


def cand_overnight_mom_20(panel_):
    def f(df):
        on = df["open"] / df["close"].shift(1) - 1.0
        return (1 + on).rolling(20).apply(np.prod, raw=True) - 1.0
    return per_asset_ohlcv(f)


CANDIDATES = {
    "usdcny_beta_cond_60x20": cand_usdcny_beta_cond,
    "dxy_beta_cond_60x20": cand_dxy_beta_cond,
    "vol_price_corr_20": cand_vol_price_corr_20,
    "body_ratio_20": cand_body_ratio_20,
    "sharpe_60": cand_sharpe_60,
    "gap_freq_20": cand_gap_freq_20,
    "tail_ratio_20": cand_tail_ratio_20,
    "drawup_60": cand_drawup_60,
    "downside_beta_ratio_60": cand_downside_beta_ratio_60,
    "overnight_mom_20": cand_overnight_mom_20,
}


# ---------------- validation ----------------
def fwd(h):
    cols = {}
    for a in WATCH:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def rank_ic(factor, fwd_df):
    ics = {}
    idx = factor.index.intersection(fwd_df.index)
    for d in idx:
        f = factor.loc[d].dropna()
        r = fwd_df.loc[d].reindex(f.index).dropna()
        if len(r) < MIN_ASSETS:
            continue
        ics[d] = spearmanr(f.reindex(r.index), r)[0]
    return pd.Series(ics).sort_index()


def stacked_corr(cand, libs):
    out = {}
    f = cand.stack().rename("f")
    for fid, ls in libs.items():
        g = ls.stack().rename("g")
        j = pd.concat([f, g], axis=1).dropna()
        if len(j) < 100:
            out[fid] = float("nan")
            continue
        r = j["f"].corr(j["g"], method="spearman")
        out[fid] = float(r) if np.isfinite(r) else float("nan")
    return out


def per_date_corr_mean(cand, libs, max_dates=700):
    out = {}
    common = cand.index.intersection(panel.index)[-max_dates:]
    for fid, ls in libs.items():
        cs = []
        for dt in common:
            if dt not in cand.index or dt not in ls.index:
                continue
            f = cand.loc[dt]
            g = ls.loc[dt]
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            m = m.reindex(f.index).fillna(False)
            if int(m.sum()) >= MIN_ASSETS:
                cs.append(spearmanr(f[m], g[m])[0])
        out[fid] = float(np.mean(cs)) if cs else float("nan")
    return out


def turnover_10d(factor):
    ranks = factor.rank(axis=1)
    out = []
    for i in range(10, len(ranks)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) < MIN_ASSETS:
            continue
        out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


def validate(name, factor, libs, libs_perdate):
    factor_w = factor.loc[:FACTOR_LAST]
    res = {"n_dates": int(factor_w.shape[0])}
    fwd10 = fwd(10)
    ic = rank_ic(factor_w, fwd10)
    direction = 1.0 if ic.mean() >= 0 else -1.0
    res["ic_h10"] = float(direction * ic.mean())
    res["icir_h10"] = float(direction * ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
    res["hit_h10"] = float((direction * ic > 0).mean()) if len(ic) else float("nan")
    res["n_h10"] = int(len(ic))
    res["direction"] = direction
    valid = factor_w.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d(factor_w)
    sc = stacked_corr(factor_w, libs)
    pc = per_date_corr_mean(factor_w, libs_perdate)
    res["max_abs_library_correlation"] = max((abs(v) for v in sc.values()), default=float("nan"))
    res["max_abs_library_corr_perdate"] = max((abs(v) for v in pc.values()), default=float("nan"))
    res["library_corrs"] = {k: round(v, 3) for k, v in sorted(sc.items(), key=lambda kv: -abs(kv[1]))}
    res["library_corrs_perdate"] = {k: round(v, 3) for k, v in sorted(pc.items(), key=lambda kv: -abs(kv[1]))}
    gate = abs(res["ic_h10"]) >= ADMISSION["ic"] and abs(res["icir_h10"]) >= ADMISSION["icir"]
    lowcorr = res["max_abs_library_correlation"] < 0.5 and res["max_abs_library_corr_perdate"] < 0.5
    res["PASS"] = bool(gate and lowcorr)
    print(f"=== {name} === dates={res['n_dates']} direction={direction:+.2f}")
    print(f"  h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} hit={res['hit_h10']:.3f} n={res['n_h10']}")
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.2f}")
    print(f"  max_lib_corr stacked={res['max_abs_library_correlation']:.3f} perdate={res['max_abs_library_corr_perdate']:.3f}")
    print(f"  corrs={res['library_corrs']}")
    print(f"  corrs_perdate={res['library_corrs_perdate']}")
    print(f"  -> {'PASS' if res['PASS'] else 'FAIL'} (gate:{gate} corr:{lowcorr})")
    print()
    return res


if __name__ == "__main__":
    libs = library_signals_full()
    print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets, warm-up through {FACTOR_LAST}")
    print(f"library factors ({len(libs)}): {sorted(libs.keys())}")
    results = {}
    for name, fn in CANDIDATES.items():
        try:
            factor = fn(panel)
            results[name] = validate(name, factor, libs, libs)
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e} ===")
    print("\n===== SUMMARY =====")
    for name, r in results.items():
        print(f"{name:<28} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
              f"maxcorr={r['max_abs_library_correlation']:.3f}/{r['max_abs_library_corr_perdate']:.3f} "
              f"cov={r['coverage_dates_ge8']:.2f} -> {'PASS' if r['PASS'] else 'FAIL'}")
    import json
    json.dump(results, open("scripts/miner_2_cycle8_results.json", "w"), indent=1, default=str)
    print("saved scripts/miner_2_cycle8_results.json")
