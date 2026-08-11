"""miner_2 cycle-2 exploration (vectorized): price-action structure, range-based vol,
volume-price, risk-adjusted momentum, autocorrelation, RSI. 15-instrument cross-asset universe.
IC = cross-sectional Spearman rank IC per date (>=8 assets). Admission: |IC|>=0.007,
|ICIR|>=0.084 @ h=10. Factor window: 2020-01-01..2026-07-15 (visible data thru 2026-07-29).
"""
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_2_lib import (load_panel, load_macro, WATCH, MIN_ASSETS, ADMISSION,
                         FACTOR_LAST, MAX_VISIBLE, fwd_returns, rank_ic_series,
                         turnover_10d_rank)

EPS = 1e-12
panel = load_panel()
rets = panel.pct_change()
mac = load_macro()


def load_ohlc():
    frames = {k: {} for k in ["open", "high", "low", "close", "volume"]}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        for k in frames:
            frames[k][s] = df[k].astype(float)
    return {k: pd.DataFrame(v, index=panel.index) for k, v in frames.items()}


F = load_ohlc()
O, H, L, C, V = F["open"], F["high"], F["low"], F["close"], F["volume"]
hl_range = (H - L).clip(lower=EPS)

# ---------------- candidate factors (vectorized) ----------------
def park_vol_20():
    # Parkinson volatility: sqrt( mean( ln(H/L)^2 ) / (4 ln 2) ), 20d
    lhl = np.log(H / L.clip(lower=EPS))
    return np.sqrt(lhl.pow(2).rolling(20, min_periods=10).mean() / (4 * np.log(2)))


def close_pos_20():
    return ((C - L) / hl_range).rolling(20, min_periods=10).mean()


def body_ratio_20():
    return ((C - O).abs() / hl_range).rolling(20, min_periods=10).mean()


def upper_wick_20():
    return ((H - np.maximum(O, C)) / hl_range).rolling(20, min_periods=10).mean()


def vol_price_corr_20():
    # corr(ret, volume) over 20d; vectorized via rolling cov/var on z-scored series
    rv = (rets - rets.rolling(20, min_periods=10).mean()) / rets.rolling(20, min_periods=10).std()
    vv = (V - V.rolling(20, min_periods=10).mean()) / V.rolling(20, min_periods=10).std()
    return (rv * vv).rolling(20, min_periods=10).mean()


def sharpe_60():
    mu = rets.rolling(60, min_periods=30).mean()
    sd = rets.rolling(60, min_periods=30).std()
    return mu / sd.clip(lower=EPS)


def autocorr_10():
    # lag-1 autocorrelation of daily returns over 10d window: cov(r,r.shift)/var(r)
    r1 = rets.shift(1)
    cov = (rets - rets.rolling(10, min_periods=6).mean()) * (r1 - r1.rolling(10, min_periods=6).mean())
    cov = cov.rolling(10, min_periods=6).mean()
    var = rets.rolling(10, min_periods=6).var()
    return cov / var.clip(lower=EPS)


def rsi_14():
    up = rets.clip(lower=0).rolling(14, min_periods=8).mean()
    dn = (-rets.clip(upper=0)).rolling(14, min_periods=8).mean()
    rs = up / dn.clip(lower=EPS)
    return 100 - 100 / (1 + rs)


CANDIDATES = {
    "park_vol_20": park_vol_20,
    "close_pos_20": close_pos_20,
    "body_ratio_20": body_ratio_20,
    "upper_wick_20": upper_wick_20,
    "vol_price_corr_20": vol_price_corr_20,
    "sharpe_60": sharpe_60,
    "autocorr_10": autocorr_10,
    "rsi_14": rsi_14,
}

# ---------------- extended library signals (all 9 effective factors) ----------
def library_signals_all():
    libs = {}
    libs["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
    libs["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
    libs["rel_mom_20d_skip5"] = (panel.shift(5) / panel.shift(25) - 1.0).sub(
        (panel.shift(5) / panel.shift(25) - 1.0).median(axis=1), axis=0)
    libs["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    libs["max_ret_20d"] = rets.rolling(20).max()
    ds = rets.where(rets < 0)
    libs["downside_vol_ratio_20"] = -(ds.rolling(20, min_periods=10).std() / rets.rolling(20, min_periods=10).std())
    libs["amihud_20"] = (rets.abs() / V.clip(lower=EPS)).rolling(20, min_periods=10).mean()
    mkt = rets.mean(axis=1)
    cov = rets.rolling(60, min_periods=30).cov(mkt)
    var = mkt.rolling(60, min_periods=30).var().clip(lower=EPS)
    libs["beta_ew_60d"] = cov.div(var, axis=0)
    try:
        vix = mac["VIX"]
        vixr = vix.pct_change()
        beta = rets.rolling(60, min_periods=30).cov(vixr) / vixr.rolling(60, min_periods=30).var().clip(lower=EPS)
        libs["vix_beta_cond_60x20"] = -beta * (vix / vix.shift(20) - 1.0)
    except Exception as e:
        print("vix lib warn:", e)
    return libs


LIBS = library_signals_all()


def library_corr_ext(factor: pd.DataFrame, max_dates=400):
    per = {}
    common = factor.index.intersection(panel.index)[-max_dates:]
    for fid, lf in LIBS.items():
        cs = []
        for dt in common:
            f = factor.loc[dt]
            g = lf.loc[dt]
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            if int(m.sum()) >= MIN_ASSETS:
                cs.append(pd.Series(f[m]).corr(pd.Series(g[m]), method="spearman"))
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per


def validate(name, factor, horizons=(1, 2, 3, 5, 10, 20)):
    fw = factor.loc[:FACTOR_LAST]
    fwd = {h: fwd_returns(panel, h) for h in horizons}
    ic_by_h = {h: rank_ic_series(fw, fwd[h]) for h in horizons}
    ic10 = ic_by_h[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    res = {"name": name, "direction": direction, "n_dates": len(ic10)}
    for h in horizons:
        ic = ic_by_h[h] * direction
        icir = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = icir
        res[f"hit_h{h}"] = float((ic > 0).mean())
    valid = fw.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank(fw)
    res["max_abs_library_correlation"], per = library_corr_ext(fw)
    res["library_corrs"] = per
    gate_ic = abs(res["ic_h10"]) >= ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= ADMISSION["icir"]
    res["pass"] = bool(gate_ic and gate_icir)
    print(f"=== {name} ===  direction={direction:+.3f}  dates={len(ic10)}  assets={panel.shape[1]}")
    print(f"  h10: IC={res['ic_h10']:+.4f}  ICIR={res['icir_h10']:+.4f}  hit={res['hit_h10']:.3f}  n={res['n_dates']}")
    print(f"  coverage_asset_days={res['coverage_asset_days']:.3f}  cov_dates_ge8={res['coverage_dates_ge8']:.3f}  "
          f"turnover={res['turnover_10d_rank']:.3f}")
    print(f"  max_abs_lib_corr={res['max_abs_library_correlation']:.3f}  per={per}")
    print(f"  decay: " + ", ".join(f"h{h}={res[f'ic_h{h}']:+.4f}" for h in horizons))
    print(f"  ADMISSION |IC|={abs(res['ic_h10']):.4f}(>=0.007:{gate_ic})  "
          f"|ICIR|={abs(res['icir_h10']):.4f}(>=0.084:{gate_icir}) -> {'PASS' if res['pass'] else 'FAIL'}")
    print()
    return res


RESULTS = {}
for name, fn in CANDIDATES.items():
    try:
        fdf = fn()
        RESULTS[name] = validate(name, fdf)
        print(f"  (n_nan={int(fdf.isna().sum().sum())}, rows={len(fdf)})")
    except Exception as e:
        print(f"=== {name} === ERROR {type(e).__name__}: {e}\n")

json.dump({k: {kk: vv for kk, vv in v.items() if kk != "library_corrs"} for k, v in RESULTS.items()},
          open("scripts/miner_2_cycle2_results.json", "w"), indent=2, default=str)
print("\nSaved -> scripts/miner_2_cycle2_results.json")
