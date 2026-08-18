"""miner_2 factor screening - 2027-01-05 cycle (fully vectorized, no per-date loops).

Explore NOVEL candidate factor families against the CURRENT 8-factor library.
IC = cross-sectional Spearman rank IC per date (>=8 assets), horizon 10 primary.
Gates: |IC|>=0.007, |ICIR|>=0.084. Data visible through 2027-01-04.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MAX_VISIBLE = "2027-01-04"
FACTOR_LAST = "2027-01-04"
MIN_ASSETS = 8
ADMISSION = {"ic": 0.007, "icir": 0.084, "corr": 0.5}
HORIZONS = (1, 2, 3, 5, 10, 20)
EPS = 1e-12


def load_panel() -> pd.DataFrame:
    closes = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        closes[s] = df["close"].astype(float)
    panel = pd.concat(closes, axis=1, sort=True)
    return panel[~panel.index.duplicated(keep="last")].sort_index()


def load_ohlc() -> dict[str, pd.DataFrame]:
    out = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[s] = df[["open", "close", "high", "low", "volume"]].astype(float)
    return out


def load_macro(name: str | None = None) -> dict[str, pd.Series]:
    out = {}
    for m in (MACRO if name is None else [name]):
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[m] = df["close"].astype(float)
    return out


def fwd_returns(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def row_pearson(X, Y, min_n=MIN_ASSETS):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    valid = np.isfinite(X) & np.isfinite(Y)
    cnt = valid.sum(axis=1)
    Xv = np.where(valid, X, np.nan)
    Yv = np.where(valid, Y, np.nan)
    Xc = Xv - np.nanmean(Xv, axis=1, keepdims=True)
    Yc = Yv - np.nanmean(Yv, axis=1, keepdims=True)
    num = np.nansum(Xc * Yc, axis=1)
    dx = np.sqrt(np.nansum(Xc * Xc, axis=1))
    dy = np.sqrt(np.nansum(Yc * Yc, axis=1))
    r = np.full(len(X), np.nan)
    m = (cnt >= min_n) & (dx > 0) & (dy > 0)
    r[m] = num[m] / (dx[m] * dy[m])
    return r


def rank_ic_series(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    common = factor.index.intersection(fwd.index)
    Fr = factor.loc[common].rank(axis=1)
    Rr = fwd.loc[common].rank(axis=1)
    return pd.Series(row_pearson(Fr.values, Rr.values), index=common).sort_index()


def turnover_10d_rank(factor: pd.DataFrame) -> float:
    ranks = factor.rank(axis=1)
    r10 = ranks.shift(10)
    diff = (ranks - r10).abs()
    both = ranks.notna() & r10.notna()
    mean_diff = diff.where(both).mean(axis=1)
    valid = both.sum(axis=1)
    mean_diff = mean_diff.where(valid >= MIN_ASSETS)
    return float(mean_diff.mean(skipna=True)) if mean_diff.notna().any() else float("nan")


# ---------------- library factors (current 8 effective + refs) ----------------
def library_signals(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    rets = panel.pct_change()
    ew = rets.mean(axis=1)
    mom20 = panel.shift(5) / panel.shift(25) - 1.0
    out["rel_mom_20d_skip5"] = mom20.sub(mom20.median(axis=1), axis=0)
    cov = rets.rolling(60).cov(ew)
    var = ew.rolling(60).var()
    out["beta_ew_60d"] = cov.div(var, axis=0)
    neg = rets.where(rets < 0, 0.0)
    down = np.sqrt((neg ** 2).rolling(20).mean())
    tot = rets.rolling(20).std()
    out["downside_vol_ratio_20"] = -(down / (tot + EPS))
    out["max_ret_20d"] = rets.rolling(20).max()
    eur = load_macro("EURUSD")["EURUSD"]
    eurr = eur.pct_change()
    b = rets.rolling(60).cov(eurr) / (eurr.rolling(60).var() + EPS)
    out["eurusd_beta_cond_60x20"] = b * (eur / eur.shift(20) - 1.0)
    corrs = {}
    for a in panel.columns:
        s = rets[a]
        others = [c for c in panel.columns if c != a]
        corrs[a] = s.rolling(60).corr(rets[others].mean(axis=1))
    out["corr_ew_60"] = pd.DataFrame(corrs, index=panel.index)
    out["kurt_20d_skip5"] = rets.rolling(20).kurt().shift(5)
    dxy = load_macro("DXY")["DXY"]
    dxy_r = dxy.pct_change()
    b = rets.rolling(60).cov(dxy_r) / (dxy_r.rolling(60).var() + EPS)
    out["dxy_beta_cond_60x20"] = -b * (dxy / dxy.shift(20) - 1.0)
    return out


def library_corr(factor: pd.DataFrame, libs: dict, max_dates: int = 700) -> tuple:
    per = {}
    for fid, lf in libs.items():
        common = factor.index.intersection(lf.index)
        if len(common) == 0:
            per[fid] = None
            continue
        common = common[-max_dates:]
        Fr = factor.loc[common].rank(axis=1)
        Gr = lf.loc[common].rank(axis=1)
        cs = row_pearson(Fr.values, Gr.values)
        cs = cs[np.isfinite(cs)]
        per[fid] = round(float(np.mean(cs)), 4) if len(cs) else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


# ---------------- candidate factor builders (novel families) ----------------
def build_candidates(panel: pd.DataFrame, ohlc: dict, macro: dict) -> dict[str, pd.DataFrame]:
    rets = panel.pct_change()
    idx = panel.index
    cands = {}

    # 1. range_pos_20: where close sits in 20d high-low range (trend location)
    hi = panel.rolling(20).max()
    lo = panel.rolling(20).min()
    cands["range_pos_20"] = (panel - lo) / (hi - lo + EPS)

    # 2. trend_r2_60: R^2 of 60d linear trend of log price (trend quality)
    lp = np.log(panel)
    t = np.arange(len(panel))
    tser = pd.Series(t, index=idx)
    lp60 = lp.rolling(60)
    slope = lp60.cov(tser) / tser.rolling(60).var()
    fit = slope * tser + lp60.mean()
    resid_var = ((lp - fit) ** 2).rolling(60).mean()
    tot_var = lp60.var()
    cands["trend_r2_60"] = 1.0 - resid_var / (tot_var + EPS)

    # 3. drawdown_60: distance from 60d high (drawdown depth, negative)
    cands["drawdown_60"] = panel / panel.rolling(60).max() - 1.0

    # 4. volume_ratio_5x20: short/medium volume attention
    vol = {a: ohlc[a]["volume"] for a in WATCH}
    voldf = pd.DataFrame(vol, index=idx)
    cands["volume_ratio_5x20"] = voldf.rolling(5).mean() / (voldf.rolling(20).mean() + EPS)

    # 5. volume_z_60: volume deviation from 60d level
    vm = voldf.rolling(60).mean()
    vs = voldf.rolling(60).std()
    cands["volume_z_60"] = (voldf - vm) / (vs + EPS)

    # 6. bond_spread_beta_cond_60x20: beta to (US10Y-CN10Y) spread * 20d spread move
    spread = panel["US10Y"] - panel["CN10Y"]
    spr = spread.pct_change()
    b = rets.rolling(60).cov(spr) / (spr.rolling(60).var() + EPS)
    cands["bond_spread_beta_cond_60x20"] = b * (spread / spread.shift(20) - 1.0)

    # 7. gold_copper_ratio_beta_cond_60x20: risk-appetite ratio conditional beta
    gc = panel["XAU"] / panel["COPPER"]
    gcr = gc.pct_change()
    b = rets.rolling(60).cov(gcr) / (gcr.rolling(60).var() + EPS)
    cands["gold_copper_beta_cond_60x20"] = b * (gc / gc.shift(20) - 1.0)

    # 8. cny_beta_cond_60x20: USDCNY (risk-off FX) conditional beta
    cny = macro["USDCNY"]
    cny_r = cny.pct_change()
    b = rets.rolling(60).cov(cny_r) / (cny_r.rolling(60).var() + EPS)
    cands["cny_beta_cond_60x20"] = b * (cny / cny.shift(20) - 1.0)

    # 9. win_rate_60_skip5: 60d win rate
    up = (rets > 0).astype(float)
    cands["win_rate_60_skip5"] = up.rolling(60).mean().shift(5)

    # 10. updown_asym_60: 60d upside/downside capture ratio
    upr = rets.where(rets > 0, np.nan).rolling(60).mean()
    dnr = rets.where(rets < 0, np.nan).rolling(60).mean()
    cands["updown_asym_60"] = upr / (dnr.abs() + EPS)

    # 11. autocorr_5d_20: 5d-lag autocorrelation over 20d window
    cands["autocorr_5d_20"] = rets.rolling(20).corr(rets.shift(5))

    # 12. range_ratio_20: average daily range vs 20d close (intraday vol share)
    ranges = {}
    for a in WATCH:
        o = ohlc[a]
        ranges[a] = (o["high"] - o["low"]) / (o["close"] + EPS)
    rng = pd.DataFrame(ranges, index=idx)
    cands["range_ratio_20"] = rng.rolling(20).mean() / (panel.rolling(20).std() + EPS)

    return cands


def validate(name: str, factor: pd.DataFrame, panel: pd.DataFrame,
             fwd: dict, libs: dict) -> dict:
    factor_w = factor.loc["2020-01-01":FACTOR_LAST]
    res = {"name": name, "factor_rows": int(factor_w.notna().sum().sum()),
           "n_assets": panel.shape[1]}
    ic10s = rank_ic_series(factor_w, fwd[10])
    direction = float(np.sign(ic10s.mean())) if np.isfinite(ic10s.mean()) and ic10s.mean() != 0 else 1.0
    res["direction"] = direction
    for h in HORIZONS:
        ic = rank_ic_series(factor_w, fwd[h]) * direction
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean()) if len(ic) else float("nan")
        res[f"n_dates_h{h}"] = int(len(ic))
    valid = factor_w.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank(factor_w)
    max_corr, per = library_corr(factor_w, libs)
    res["max_abs_library_correlation"] = max_corr
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in HORIZONS}
    gate_ic = abs(res["ic_h10"]) >= ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= ADMISSION["icir"]
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "pass": bool(gate_ic and gate_icir)}
    res["PASS"] = bool(gate_ic and gate_icir)
    return res


def main():
    panel = load_panel()
    ohlc = load_ohlc()
    macro = load_macro()
    print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets, "
          f"{panel.index.min().date()} .. {panel.index.max().date()}", flush=True)
    fwd = {h: fwd_returns(panel, h) for h in HORIZONS}
    libs = library_signals(panel)
    cands = build_candidates(panel, ohlc, macro)
    results = {}
    for name, factor in cands.items():
        r = validate(name, factor, panel, fwd, libs)
        results[name] = r
        print(f"=== {name} ===", flush=True)
        print(f"  direction={r['direction']:+.3f}", flush=True)
        for h in HORIZONS:
            print(f"  h{h:>2}: IC={r[f'ic_h{h}']:+.4f}  ICIR={r[f'icir_h{h}']:+.4f}  "
                  f"hit={r[f'hit_h{h}']:.3f}  n={r[f'n_dates_h{h}']}", flush=True)
        print(f"  coverage_asset_days={r['coverage_asset_days']:.3f}  "
              f"coverage_dates_ge8={r['coverage_dates_ge8']:.3f}  "
              f"turnover_10d_rank={r['turnover_10d_rank']:.3f}", flush=True)
        print(f"  max_abs_library_corr={r['max_abs_library_correlation']:.3f}  per={r['library_corrs']}", flush=True)
        print(f"  ADMISSION (h=10): |IC|={abs(r['ic_h10']):.4f} (>=0.007: {r['admission_gate']['ic_pass']}), "
              f"|ICIR|={abs(r['icir_h10']):.4f} (>=0.084: {r['admission_gate']['icir_pass']}) -> "
              f"{'PASS' if r['PASS'] else 'FAIL'}", flush=True)
        print(flush=True)
    with open("scripts/miner_2_20270105_screen_results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("SAVED scripts/miner_2_20270105_screen_results.json", flush=True)


if __name__ == "__main__":
    main()
