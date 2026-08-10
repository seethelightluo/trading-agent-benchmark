"""miner_1 factor exploration v2 - NEW candidate factors on the 15-asset cross-asset universe.

Uses the shared factor_validate framework (file-based panel, visible window only).
Admission gate (shared): |IC10| >= 0.0070 and |ICIR10| >= 0.0840.
Also reports h=1 daily IC for context. Each candidate is a SINGLE interpretable idea.
"""
import json, os, sys, base64, gzip, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (
    closes_panel, macro_closes, ic_series, forward_returns,
    summary_metrics, regime_split, library_ic_series_map, max_abs_library_corr,
)

VISIBLE = "2026-07-29"
close = closes_panel(VISIBLE)
macro = macro_closes(VISIBLE)
ret = close.pct_change()

# ---------------- candidate builders (each returns DataFrame dates x assets) ----------------
def cand_mom_vol_scaled(close, ret, **p):
    """quality momentum: 20d momentum (10d skip) scaled by inverse 20d vol."""
    mom = close.shift(p["skip"]) / close.shift(p["lookback"] + p["skip"]) - 1.0
    vol = ret.rolling(p["vol_win"], min_periods=p["vol_win"] // 2).std()
    return mom / vol

def cand_breadth_cond_mom(close, ret, **p):
    """market-breadth conditional momentum: asset 20d mom x fraction of universe with positive 20d return."""
    mom = close / close.shift(p["mom_win"]) - 1.0
    breadth = (mom > 0).sum(axis=1) / mom.notna().sum(axis=1)
    return mom.mul(breadth, axis=0)

def cand_range_pos(close, ret=None, macro=None, **p):
    """intraday position: mean of (close-low)/(high-low) over win days (buying pressure)."""
    frames = {}
    for s in close.columns:
        df = pd.read_csv(os.path.join("../persistent/stock_data", s + ".csv"), parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE)].set_index("date")
        rng = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
        frames[s] = rng.rolling(p["win"], min_periods=p["win"] // 2).mean()
    return pd.DataFrame(frames).reindex(close.index)

def cand_yield_spread_beta(close, ret, macro, **p):
    """cross-border rate carry: beta(asset_ret, d(US10Y-CN10Y)) x spread 20d change."""
    spread = close["US10Y"] - close["CN10Y"]
    dspread = spread.diff()
    smom = spread / spread.shift(p["mwin"]) - 1.0
    out = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), dspread.rename("d")], axis=1).dropna()
        b = pair["a"].rolling(p["beta_win"], min_periods=p["beta_win"] // 2).cov(pair["d"]) / \
            pair["d"].rolling(p["beta_win"], min_periods=p["beta_win"] // 2).var()
        out[a] = b * smom
    return pd.DataFrame(out).reindex(close.index)

def cand_btc_eth_ratio_beta(close, ret, macro, **p):
    """crypto risk-on gauge: beta(asset_ret, d(log BTC/ETH)) x BTC/ETH ratio momentum."""
    ratio = close["BTC"] / close["ETH"]
    rret = np.log(ratio).diff()
    rmom = ratio / ratio.shift(p["mwin"]) - 1.0
    out = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), rret.rename("r")], axis=1).dropna()
        b = pair["a"].rolling(p["beta_win"], min_periods=p["beta_win"] // 2).cov(pair["r"]) / \
            pair["r"].rolling(p["beta_win"], min_periods=p["beta_win"] // 2).var()
        out[a] = b * rmom
    return pd.DataFrame(out).reindex(close.index)

def cand_vix_delta_beta(close, ret, macro, **p):
    """VIX delta beta: -beta(asset, VIX_ret, 60) x (VIX_10d_change / VIX_level)."""
    vix = macro["VIX"]
    vix_ret = vix.pct_change()
    vdelta = vix / vix.shift(p["mwin"]) - 1.0
    out = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), vix_ret.rename("v")], axis=1).dropna()
        b = pair["a"].rolling(p["beta_win"], min_periods=p["beta_win"] // 2).cov(pair["v"]) / \
            pair["v"].rolling(p["beta_win"], min_periods=p["beta_win"] // 2).var()
        out[a] = -b * vdelta
    return pd.DataFrame(out).reindex(close.index)

def cand_vol_ratio(close, ret, **p):
    """short/long vol ratio: vol compression/expansion signal."""
    sv = ret.rolling(p["short"], min_periods=p["short"] // 2).std()
    lv = ret.rolling(p["long"], min_periods=p["long"] // 2).std()
    return sv / lv

def cand_dist_ma(close, **p):
    """distance from long moving average (trend risk premium)."""
    ma = close.rolling(p["win"], min_periods=p["win"] // 2).mean()
    return close / ma - 1.0

def cand_skew_vol_comp(close, ret, **p):
    """crash-risk combo: 20d skew x vol ratio 10x60 (negative skew + vol expansion)."""
    sk = ret.rolling(p["skew_win"], min_periods=p["skew_win"] // 2).skew()
    vr = ret.rolling(10, min_periods=5).std() / ret.rolling(60, min_periods=30).std()
    return sk * vr

def cand_eff_ratio(close, ret, **p):
    """trend efficiency: |net move| / path length over win."""
    num = (close - close.shift(p["win"])).abs()
    den = ret.abs().rolling(p["win"], min_periods=p["win"] // 2).sum()
    return num / den

def cand_cn_rate_beta(close, ret, macro, **p):
    """China rate regime: beta(asset_ret, dCN10Y) x CN10Y 20d change."""
    cn = close["CN10Y"]
    dcn = cn.diff()
    cmom = cn / cn.shift(p["mwin"]) - 1.0
    out = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), dcn.rename("d")], axis=1).dropna()
        b = pair["a"].rolling(p["beta_win"], min_periods=p["beta_win"] // 2).cov(pair["d"]) / \
            pair["d"].rolling(p["beta_win"], min_periods=p["beta_win"] // 2).var()
        out[a] = b * cmom
    return pd.DataFrame(out).reindex(close.index)

CANDIDATES = {
    "mom_vol_scaled_20x10": (cand_mom_vol_scaled, {"lookback": 20, "skip": 10, "vol_win": 20}),
    "breadth_cond_mom_20":  (cand_breadth_cond_mom, {"mom_win": 20}),
    "range_pos_10":         (cand_range_pos, {"win": 10}),
    "yield_spread_beta_60x20": (cand_yield_spread_beta, {"beta_win": 60, "mwin": 20}),
    "btc_eth_ratio_beta_60x20": (cand_btc_eth_ratio_beta, {"beta_win": 60, "mwin": 20}),
    "vix_delta_beta_60x10": (cand_vix_delta_beta, {"beta_win": 60, "mwin": 10}),
    "vol_ratio_5x60":       (cand_vol_ratio, {"short": 5, "long": 60}),
    "dist_ma120":           (cand_dist_ma, {"win": 120}),
    "skew_vol_comp_20":     (cand_skew_vol_comp, {"skew_win": 20}),
    "eff_ratio_60":         (cand_eff_ratio, {"win": 60}),
    "cn_rate_beta_60x20":   (cand_cn_rate_beta, {"beta_win": 60, "mwin": 20}),
}

def main():
    fr10 = forward_returns(close, 10)
    lib_ics = library_ic_series_map(close, h=10)  # active library (should be empty -> no corr)

    results = {}
    if os.path.exists("scripts/miner_1_20260730_explore_v2_results.json"):
        results = json.load(open("scripts/miner_1_20260730_explore_v2_results.json"))
    for fid, (fn, params) in CANDIDATES.items():
        if fid in results:
            print(f"[done] {fid} already in results")
            continue
        print(f"...computing {fid}", flush=True)
        try:
            sig = fn(close, ret, macro=macro, **params)
        except TypeError:
            try:
                sig = fn(close, ret, **params)
            except TypeError:
                sig = fn(close, **params)
        sig = sig.reindex(close.index)
        ic10 = ic_series(sig, fr10, min_valid=8)
        m10 = summary_metrics(ic10, sig, fr10, close, h=10)
        if m10 is None:
            print(f"[SKIP] {fid}: insufficient IC dates")
            continue
        # h=1 context (cheap: mean only)
        fr1 = forward_returns(close, 1)
        ic1 = ic_series(sig, fr1, min_valid=8)
        m1 = {"ic": float(ic1.mean()) if len(ic1) else None,
              "icir": float(ic1.mean() / ic1.std(ddof=1)) if len(ic1) > 2 and ic1.std(ddof=1) > 0 else None}
        rho = max_abs_library_corr(ic10, lib_ics)
        regimes = regime_split(ic10)
        gate = (abs(m10["ic"]) >= 0.0070 and abs(m10["icir"]) >= 0.0840)
        results[fid] = {
            "ic10": m10["ic"], "icir10": m10["icir"], "hit10": m10["ic_hit_ratio"],
            "n10": m10["n_ic_dates"], "cov_ad": m10["coverage_asset_days"],
            "cov_d8": m10["coverage_dates_ge8"], "turn": m10["turnover_10d_rank"],
            "ic1": m1["ic"] if m1 else None, "icir1": m1["icir"] if m1 else None,
            "decay": m10["decay_ic_by_horizon"], "max_rho_lib": rho,
            "regimes": regimes, "gate_pass": gate,
        }
        flag = "PASS" if gate else "fail"
        print(f"[{flag}] {fid}: IC10={m10['ic']:+.4f} ICIR10={m10['icir']:+.3f} hit={m10['ic_hit_ratio']:.3f} "
              f"n={m10['n_ic_dates']} cov_ad={m10['coverage_asset_days']:.3f} cov_d8={m10['coverage_dates_ge8']:.3f} "
              f"turn={m10['turnover_10d_rank'] if m10['turnover_10d_rank'] else float('nan'):.2f} rho_lib={rho:.3f} ic1={m1['ic'] if m1 else float('nan'):+.4f}")
        print(f"      decay(1,2,3,5,10,20)={[m10['decay_ic_by_horizon'][h] for h in ('1','2','3','5','10','20')]}")
        print(f"      regimes={regimes}")
        # stash panel for later persistence
        os.makedirs("scripts/_panels", exist_ok=True)
        sig.round(8).to_csv(f"scripts/_panels/{fid}.csv")

    with open("scripts/miner_1_20260730_explore_v2_results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("saved exploration results.")

if __name__ == "__main__":
    main()
