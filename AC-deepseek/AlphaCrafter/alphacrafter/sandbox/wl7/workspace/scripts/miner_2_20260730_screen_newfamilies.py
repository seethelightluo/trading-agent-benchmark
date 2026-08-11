"""miner_2 cycle 6: screen NEW factor families not yet covered in the library.
Universe: 15 tradable cross-asset instruments. Validation window 2020-01-01..2026-07-15.
Gate: |IC|>=0.007 and |ICIR|>=0.084 @ h=10 (benchmark contract for this universe).
Families: price-range location, 52w/6m high proximity, overnight/intraday return
decomposition, return-volume correlation, downside semideviation, drawdown depth,
vol-scaled short-term reversal.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner_2_lib as lib

panel = lib.load_panel()
macro = lib.load_macro()
rets = panel.pct_change()
eps = 1e-12

# ---------- candidate factor builders (per-asset calendar where lags matter) ----------

def cand_near_high_120(p, m):
    return p / p.rolling(120, min_periods=60).max() - 1.0

def cand_near_high_250(p, m):
    return p / p.rolling(250, min_periods=120).max() - 1.0

def cand_up_day_ratio_60(p, m):
    r = p.pct_change()
    return (r > 0).rolling(60, min_periods=30).mean()

def cand_range_pos_20(p, m):
    # per-asset calendar intraday close location within daily range, 20d mean
    def f(s):
        df = s.to_frame("close")
        # need open/high/low: fetch raw per asset
        return None
    return None  # handled separately (needs OHLC)

def cand_range_pos_20_full(p, m):
    # position of close within 20d high-low range (stochastic %K style)
    hi = p.rolling(20, min_periods=10).max()
    lo = p.rolling(20, min_periods=10).min()
    return (p - lo) / (hi - lo + eps)

def cand_corr_ret_vol_20(p, m):
    # rolling correlation of returns with log volume (per-asset calendar)
    def f(s):
        df = s.to_frame("close")
        df["vol"] = None
        return None
    return None  # handled separately (needs volume)

def cand_downside_dev_20(p, m):
    r = p.pct_change()
    neg = r.where(r < 0, 0.0)
    return (neg.pow(2)).rolling(20, min_periods=10).mean().apply(np.sqrt)

def cand_max_dd_60(p, m):
    dd = p / p.rolling(60, min_periods=30).max() - 1.0
    return dd.rolling(60, min_periods=30).min()

def cand_rev_5d_vol(p, m):
    r5 = p / p.shift(5) - 1.0
    v20 = rets.rolling(20, min_periods=10).std()
    return -(r5 / (v20 + eps))

def cand_up_down_ratio_20(p, m):
    r = p.pct_change()
    up = r.where(r > 0, np.nan).rolling(20, min_periods=10).mean()
    dn = r.where(r < 0, np.nan).rolling(20, min_periods=10).mean()
    return up / (dn.abs() + eps)

def cand_overnight_ret_20(p, m):
    # needs open: handled separately
    return None

def cand_intraday_ret_20(p, m):
    # needs open: handled separately
    return None


# ---------- OHLC/volume-based candidates (need open/high/low/volume) ----------

def load_ohlc_panel():
    closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
    for s in lib.WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= lib.MAX_VISIBLE].set_index("date").sort_index()
        closes[s] = df["close"].astype(float)
        opens[s] = df["open"].astype(float)
        highs[s] = df["high"].astype(float)
        lows[s] = df["low"].astype(float)
        vols[s] = df["volume"].astype(float)
    return (pd.concat(closes, axis=1).sort_index(), pd.concat(opens, axis=1).sort_index(),
            pd.concat(highs, axis=1).sort_index(), pd.concat(lows, axis=1).sort_index(),
            pd.concat(vols, axis=1).sort_index())


def cand_range_pos_20_ohlc(close, open_, high, low, vol, panel):
    # per-asset calendar: mean of (close-low)/(high-low) over 20d
    out = {}
    for s in panel.columns:
        h, l, c = high[s].dropna(), low[s].dropna(), close[s].dropna()
        idx = c.index.intersection(h.index).intersection(l.index)
        pos = (c[idx] - l[idx]) / (h[idx] - l[idx] + eps)
        out[s] = pos.rolling(20, min_periods=10).mean()
    return pd.DataFrame(out, index=panel.index)

def cand_overnight_ret_20(close, open_, high, low, vol, panel):
    out = {}
    for s in panel.columns:
        c, o = close[s].dropna(), open_[s].dropna()
        gap = o / c.shift(1) - 1.0
        out[s] = gap.rolling(20, min_periods=10).mean()
    return pd.DataFrame(out, index=panel.index)

def cand_intraday_ret_20(close, open_, high, low, vol, panel):
    out = {}
    for s in panel.columns:
        c, o = close[s].dropna(), open_[s].dropna()
        ir = c / o - 1.0
        out[s] = ir.rolling(20, min_periods=10).mean()
    return pd.DataFrame(out, index=panel.index)

def cand_corr_ret_vol_20(close, open_, high, low, vol, panel):
    out = {}
    for s in panel.columns:
        c, v = close[s].dropna(), vol[s].dropna()
        r = c.pct_change()
        lv = np.log(v + eps)
        m = pd.concat([r, lv], axis=1).dropna()
        corr = m.iloc[:,0].rolling(20, min_periods=10).corr(m.iloc[:,1])
        out[s] = corr
    return pd.DataFrame(out, index=panel.index)

def cand_dollar_vol_trend_20(close, open_, high, low, vol, panel):
    out = {}
    for s in panel.columns:
        c, v = close[s].dropna(), vol[s].dropna()
        dv = (c * v).rolling(20, min_periods=10).mean()
        out[s] = dv / dv.shift(60) - 1.0  # 60d change in 20d avg dollar volume
    return pd.DataFrame(out, index=panel.index)


# ---------- validation ----------

horizons = (1, 2, 3, 5, 10, 20)
fwd = {h: lib.fwd_returns(panel, h) for h in horizons}

def run_validation(name, factor, direction_override=None):
    factor = factor.reindex(panel.index)
    factor_w = factor.loc[:lib.FACTOR_LAST]
    if factor_w.notna().sum().sum() < 100:
        print(f"=== {name}: insufficient data ===")
        return None
    res = {"name": name, "factor_rows": len(factor_w), "n_assets": panel.shape[1]}
    ic_by_h = {h: lib.rank_ic_series(factor_w, fwd[h]) for h in horizons}
    ic10 = ic_by_h[10]
    direction = direction_override if direction_override is not None else (
        float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0)
    ic_by_h = {h: ic * direction for h, ic in ic_by_h.items()}
    for h in horizons:
        ic = ic_by_h[h]
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean())
        res[f"n_dates_h{h}"] = int(len(ic))
    res["direction"] = direction
    valid = factor_w.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= lib.MIN_ASSETS).mean())
    res["turnover_10d_rank"] = lib.turnover_10d_rank(factor_w)
    libs = lib.library_signals(panel)
    max_corr, per = lib.library_corr(factor_w, panel, libs)
    res["max_abs_library_correlation"] = max_corr
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in horizons}
    gate_ic = abs(res["ic_h10"]) >= lib.ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= lib.ADMISSION["icir"]
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "pass": bool(gate_ic and gate_icir)}
    print(f"=== {name} ===")
    print(f"  window {factor_w.index.min().date()}..{factor_w.index.max().date()} | {len(factor_w)} dates | dir={direction:+.2f}")
    print(f"  h10: IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} hit={res['hit_h10']:.3f} n={res['n_dates_h10']}")
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_dates={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.2f}")
    print(f"  maxcorr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    print(f"  decay={res['decay_ic_by_horizon']}")
    print(f"  ADMISSION -> {'PASS' if gate_ic and gate_icir else 'FAIL'}")
    print()
    return res


C = {}
C["near_high_120"] = cand_near_high_120(panel, macro)
C["near_high_250"] = cand_near_high_250(panel, macro)
C["up_day_ratio_60"] = cand_up_day_ratio_60(panel, macro)
C["range_pos_20d"] = cand_range_pos_20_full(panel, macro)
C["downside_dev_20"] = cand_downside_dev_20(panel, macro)
C["max_dd_60"] = cand_max_dd_60(panel, macro)
C["rev_5d_vol"] = cand_rev_5d_vol(panel, macro)
C["up_down_ratio_20"] = cand_up_down_ratio_20(panel, macro)

close, open_, high, low, vol = load_ohlc_panel()
C["range_pos_ohlc_20"] = cand_range_pos_20_ohlc(close, open_, high, low, vol, panel)
C["overnight_ret_20"] = cand_overnight_ret_20(close, open_, high, low, vol, panel)
C["intraday_ret_20"] = cand_intraday_ret_20(close, open_, high, low, vol, panel)
C["corr_ret_vol_20"] = cand_corr_ret_vol_20(close, open_, high, low, vol, panel)
C["dollar_vol_trend_20x60"] = cand_dollar_vol_trend_20(close, open_, high, low, vol, panel)

RESULTS = {}
for name, f in C.items():
    try:
        RESULTS[name] = run_validation(name, f)
    except Exception as e:
        print(f"=== {name}: ERROR {type(e).__name__}: {e} ===")

print("\n===== SUMMARY (h10 gate |IC|>=0.007, |ICIR|>=0.084) =====")
for name, r in RESULTS.items():
    if r is None:
        continue
    passed = r["admission_gate"]["pass"]
    print(f"{name:<24} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"maxcorr={r['max_abs_library_correlation']:.3f} -> {'PASS' if passed else 'FAIL'}")
