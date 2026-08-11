"""miner_2 cycle-3 screen: novel macro-beta / vol-structure / price-structure factors.
Universe: 15 tradable cross-asset instruments. Window 2020-01-01..2026-07-15.
Admission (benchmark contract): |IC|>=0.007 and |ICIR|>=0.084 @ h=10.
Correlation audit vs FULL 9-factor library (mean per-date cross-sectional Spearman).
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from miner_2_lib import (load_panel, load_macro, per_asset, fwd_returns,
                         MIN_ASSETS, FACTOR_LAST, turnover_10d_rank)

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

panel = load_panel()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)


def ic_series_fast(factor, h):
    fwd = fwd_returns(panel, h)
    dates = factor.index.intersection(fwd.index)
    dates = dates[dates <= pd.Timestamp(FACTOR_LAST)]
    F = factor.reindex(dates).values.astype(float)
    R = fwd.reindex(dates).values.astype(float)
    A = np.argsort(np.argsort(F, axis=1), axis=1).astype(float)
    B = np.argsort(np.argsort(R, axis=1), axis=1).astype(float)
    out, idx = [], []
    for i in range(len(dates)):
        m = np.isfinite(F[i]) & np.isfinite(R[i])
        if int(m.sum()) < MIN_ASSETS:
            continue
        a_, b_ = A[i][m], B[i][m]
        ma, mb = a_.mean(), b_.mean()
        num = float(((a_ - ma) * (b_ - mb)).sum())
        den = float(np.sqrt(((a_ - ma) ** 2).sum() * ((b_ - mb) ** 2).sum()))
        out.append(num / den if den > 0 else 0.0)
        idx.append(dates[i])
    return pd.Series(out, index=idx)


def rolling_beta(s, m, win):
    s, m = s.dropna(), m.dropna()
    z = pd.concat([s.pct_change().rename("r"), m.pct_change().rename("m")], axis=1).dropna()
    beta = z["r"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var().replace(0, np.nan)
    return beta


def per_asset_beta(macro_series, win=60):
    def f(panel, macro):
        m = macro_series(panel, macro).dropna()
        cols = {}
        for a in panel.columns:
            cols[a] = rolling_beta(panel[a], m, win)
        return pd.DataFrame(cols, index=panel.index)
    return f


def cand_usdjpy_beta_60(panel, macro): return per_asset_beta(lambda p, m: m["USDJPY"])(panel, macro)
def cand_eurusd_beta_60(panel, macro): return per_asset_beta(lambda p, m: m["EURUSD"])(panel, macro)
def cand_usdcny_beta_60(panel, macro): return per_asset_beta(lambda p, m: m["USDCNY"])(panel, macro)
def cand_dxy_beta_60(panel, macro): return per_asset_beta(lambda p, m: m["DXY"])(panel, macro)
def cand_gold_beta_60(panel, macro): return per_asset_beta(lambda p, m: p["XAU"])(panel, macro)
def cand_btc_beta_60(panel, macro): return per_asset_beta(lambda p, m: p["BTC"])(panel, macro)
def cand_ust_beta_60(panel, macro): return per_asset_beta(lambda p, m: p["US10Y"])(panel, macro)


def cand_corr_ew_60(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        m = mkt.reindex(s.index)
        z = pd.concat([s.pct_change().rename("r"), m.pct_change().rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60).corr(z["m"])
    return pd.DataFrame(cols, index=panel.index)


def cand_idio_vol_60(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        m = mkt.reindex(s.index)
        z = pd.concat([s.pct_change().rename("r"), m.pct_change().rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var().replace(0, np.nan)
        resid = z["r"] - beta * z["m"]
        cols[a] = resid.rolling(60).std()
    return pd.DataFrame(cols, index=panel.index)


def cand_vol_term_5x60(panel, macro):
    def f(s):
        r = s.pct_change()
        return r.rolling(5).std() / r.rolling(60).std()
    return per_asset(f)(panel, macro)


def cand_updown_vol_ratio_60(panel, macro):
    def f(s):
        r = s.pct_change()
        up = r.clip(lower=0).rolling(60).std()
        dn = r.clip(upper=0).rolling(60).std()
        return up / dn.replace(0, np.nan)
    return per_asset(f)(panel, macro)


def cand_eff_ratio_60(panel, macro):
    def f(s):
        return (s - s.shift(60)).abs() / s.pct_change().abs().rolling(60).sum().replace(0, np.nan)
    return per_asset(f)(panel, macro)


def cand_vol_price_corr_60(panel, macro):
    """rolling 60d corr(|ret|, volume) -- volume-price synchronicity."""
    vols = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        v = df["volume"].astype(float).reindex(panel.index)
        r = panel[a].pct_change().abs()
        vols[a] = pd.concat([r.rename("r"), v.rename("v")], axis=1).rolling(60).corr()["r"].unstack()["v"]
    return pd.DataFrame(vols, index=panel.index)


def cand_body_ratio_60(panel, macro):
    cols = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        o, c, h, l = (df[k].astype(float).reindex(panel.index) for k in ["open", "close", "high", "low"])
        rng = (h - l).replace(0, np.nan)
        cols[a] = ((c - o).abs() / rng).rolling(60).mean()
    return pd.DataFrame(cols, index=panel.index)


def cand_usdjpy_beta_cond_60x20(panel, macro):
    jpy = macro["USDJPY"].dropna()
    def f(s):
        z = pd.concat([s.pct_change().rename("r"), jpy.pct_change().rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var().replace(0, np.nan)
        jpy20 = (jpy / jpy.shift(20) - 1.0).reindex(s.index)
        return -beta * jpy20
    return per_asset(f)(panel, macro)


def cand_eurusd_beta_cond_60x20(panel, macro):
    eu = macro["EURUSD"].dropna()
    def f(s):
        z = pd.concat([s.pct_change().rename("r"), eu.pct_change().rename("m")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var().replace(0, np.nan)
        eu20 = (eu / eu.shift(20) - 1.0).reindex(s.index)
        return beta * eu20
    return per_asset(f)(panel, macro)


# ---------- full 9-factor library for correlation audit ----------
def build_library():
    lib = {}
    mom20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(panel, macro)
    lib["rel_mom_20d_skip5"] = mom20.sub(mom20.median(axis=1), axis=0)
    lib["beta_ew_60d"] = per_asset(lambda s: rolling_beta(s, mkt, 60))(panel, macro)
    lib["mom_120d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(125) - 1.0)(panel, macro)
    lib["vol_of_vol20x60"] = per_asset(lambda s: s.pct_change().rolling(20).std().rolling(60).std())(panel, macro)
    lib["max_ret_20d"] = per_asset(lambda s: s.pct_change().rolling(20).max())(panel, macro)
    lib["downside_vol_ratio_20"] = per_asset(
        lambda s: -(s.pct_change().clip(upper=0).rolling(20).std() / s.pct_change().rolling(20).std().replace(0, np.nan)))(panel, macro)
    lib["mom_10d_skip5"] = per_asset(lambda s: s.shift(5) / s.shift(15) - 1.0)(panel, macro)
    vix = macro["VIX"].dropna()
    lib["vix_beta_cond_60x20"] = per_asset(
        lambda s: -rolling_beta(s, vix, 60) * (vix / vix.shift(20) - 1.0).reindex(s.index))(panel, macro)
    vols = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        v = df["volume"].astype(float).reindex(panel.index)
        r = panel[a].pct_change().abs()
        vols[a] = (r / v.replace(0, np.nan)).rolling(20).mean()
    lib["amihud_20"] = pd.DataFrame(vols, index=panel.index)
    return lib


def library_corr(factor, libs, max_dates=700):
    per = {}
    common = factor.index.intersection(panel.index)
    dates = common[-max_dates:]
    for fid, lf in libs.items():
        cs = []
        for dt in dates:
            f = factor.loc[dt]
            g = lf.loc[dt]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[-1]
            if isinstance(g, pd.DataFrame):
                g = g.iloc[-1]
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            m = m.reindex(f.index).fillna(False)
            if int(m.sum()) >= MIN_ASSETS:
                cs.append(spearmanr(f[m], g[m])[0])
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


CANDS = {
    "usdjpy_beta_60": cand_usdjpy_beta_60,
    "eurusd_beta_60": cand_eurusd_beta_60,
    "usdcny_beta_60": cand_usdcny_beta_60,
    "dxy_beta_60": cand_dxy_beta_60,
    "gold_beta_60": cand_gold_beta_60,
    "btc_beta_60": cand_btc_beta_60,
    "ust_beta_60": cand_ust_beta_60,
    "corr_ew_60": cand_corr_ew_60,
    "idio_vol_60": cand_idio_vol_60,
    "vol_term_5x60": cand_vol_term_5x60,
    "updown_vol_ratio_60": cand_updown_vol_ratio_60,
    "eff_ratio_60": cand_eff_ratio_60,
    "vol_price_corr_60": cand_vol_price_corr_60,
    "body_ratio_60": cand_body_ratio_60,
    "usdjpy_beta_cond_60x20": cand_usdjpy_beta_cond_60x20,
    "eurusd_beta_cond_60x20": cand_eurusd_beta_cond_60x20,
}

lib = build_library()
print(f"panel {panel.index[0].date()}..{panel.index[-1].date()}, assets={panel.shape[1]}, library={list(lib.keys())}")

results = {}
for name, fn in CANDS.items():
    try:
        factor = fn(panel, macro)
        fw = factor.loc[:FACTOR_LAST]
        ic = ic_series_fast(factor, 10)
        if len(ic) < 200:
            print(f"{name}: TOO FEW IC DATES {len(ic)}")
            continue
        direction = float(np.sign(ic.mean())) if np.isfinite(ic.mean()) and ic.mean() != 0 else 1.0
        ic_adj = ic * direction
        icir = float(ic_adj.mean() / ic_adj.std()) if ic_adj.std() > 0 else float("nan")
        hit = float((ic_adj > 0).mean())
        valid = fw.notna()
        cov_ad = float(valid.mean().mean())
        cov_d8 = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
        to = turnover_10d_rank(fw)
        maxc, per = library_corr(factor, lib)
        ic1 = ic_series_fast(factor, 1) * direction
        ic5 = ic_series_fast(factor, 5) * direction
        ic20 = ic_series_fast(factor, 20) * direction
        pass_gate = abs(float(ic_adj.mean())) >= 0.007 and abs(icir) >= 0.084
        results[name] = {"direction": direction, "ic_h10": float(ic_adj.mean()), "icir_h10": icir,
                         "hit_h10": hit, "n_dates": len(ic), "cov_ad": cov_ad, "cov_d8": cov_d8,
                         "turnover": to, "max_corr": maxc, "per": per,
                         "ic_h1": float(ic1.mean()), "ic_h5": float(ic5.mean()), "ic_h20": float(ic20.mean()),
                         "pass": pass_gate}
        print(f"=== {name} === dir={direction:+.2f} h10 IC={float(ic_adj.mean()):+.4f} "
              f"ICIR={icir:+.4f} hit={hit:.3f} n={len(ic)} cov={cov_ad:.3f}/{cov_d8:.3f} "
              f"to={to:.3f} maxcorr={maxc:.3f} h1={float(ic1.mean()):+.4f} h5={float(ic5.mean()):+.4f} "
              f"h20={float(ic20.mean()):+.4f} PASS={pass_gate}")
    except Exception as e:
        print(f"{name}: ERROR {type(e).__name__}: {e}")

import json
with open("scripts/miner_2_cycle3_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("saved scripts/miner_2_cycle3_results.json")
