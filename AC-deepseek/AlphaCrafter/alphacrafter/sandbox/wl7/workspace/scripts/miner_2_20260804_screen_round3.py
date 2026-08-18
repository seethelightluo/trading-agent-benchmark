"""miner_2 screen round 3 (2026-08-04 sim date): price/volume STRUCTURE families.
Motivation: the post-Miner gate evicts any factor with |rho| > 0.5 vs crypto_beta_60d
(q=0.0104 anchor). Beta/momentum/vol-level factors all conflict with it. This round
tests intraday-structure and flow factors that should be orthogonal to 60d beta.

Candidates (all per-asset calendar-aware, warm-up 2020-01-01..2026-07-15, h=10 gate):
 1. eff_ratio_20d_skip5  - signed Kaufman efficiency ratio (trend quality)
 2. candle_body_20d      - mean (close-open)/(high-low), intraday direction
 3. close_pos_20d        - mean (close-low)/(high-low), close position in range
 4. vol_zscore_20x120    - volume 20d mean z-score vs 120d (flow pressure)
 5. kurt_20d_skip5       - realized kurtosis of 20d returns (tail shape)
 6. shadow_ratio_20d     - mean log upper/lower shadow ratio (reversal pressure)
 7. maxmin_20d           - rolling 20d max gain / |min loss|
 8. range_cv_20d         - CV of daily (high-low)/close range (range instability)
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_lib import validate_factor, load_panel, load_macro, MAX_VISIBLE

_OHLCV = {}


def _ohlcv(sym):
    if sym in _OHLCV:
        return _OHLCV[sym]
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
    _OHLCV[sym] = df[["open", "high", "low", "volume"]].astype(float)
    return _OHLCV[sym]


def cand_eff_ratio_20d_skip5(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        mom20 = s.shift(5) / s.shift(25) - 1.0
        gross = r.abs().rolling(20, min_periods=10).sum().shift(5)
        cols[a] = mom20 / gross  # signed efficiency in [-1, 1]
    return pd.DataFrame(cols, index=panel.index)


def cand_candle_body_20d(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        hl = _ohlcv(a).reindex(s.index)
        rng = (hl["high"] - hl["low"])
        body = ((hl["close"] if "close" in hl else s) - hl["open"]) / rng.replace(0, np.nan)
        body = body.fillna(0.0)
        cols[a] = body.rolling(20, min_periods=12).mean()
    return pd.DataFrame(cols, index=panel.index)


def cand_close_pos_20d(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        hl = _ohlcv(a).reindex(s.index)
        rng = (hl["high"] - hl["low"]).replace(0, np.nan)
        pos = (s - hl["low"]) / rng
        pos = pos.fillna(0.5)
        cols[a] = pos.rolling(20, min_periods=12).mean()
    return pd.DataFrame(cols, index=panel.index)


def cand_vol_zscore_20x120(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        v = _ohlcv(a)["volume"].reindex(s.index)
        m20 = v.rolling(20, min_periods=10).mean()
        m120 = v.rolling(120, min_periods=40).mean()
        sd120 = v.rolling(120, min_periods=40).std()
        cols[a] = (m20 - m120) / sd120.replace(0, np.nan)
    return pd.DataFrame(cols, index=panel.index)


def cand_kurt_20d_skip5(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        cols[a] = r.shift(5).rolling(20, min_periods=12).kurt()
    return pd.DataFrame(cols, index=panel.index)


def cand_shadow_ratio_20d(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        hl = _ohlcv(a).reindex(s.index)
        up = (hl["high"] - np.maximum(hl["open"], s)).clip(lower=0.0)
        lo = (np.minimum(hl["open"], s) - hl["low"]).clip(lower=0.0)
        ratio = np.log((up + 1e-9) / (lo + 1e-9))
        cols[a] = ratio.rolling(20, min_periods=12).mean()
    return pd.DataFrame(cols, index=panel.index)


def cand_maxmin_20d(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        mx = r.rolling(20, min_periods=10).max()
        mn = r.rolling(20, min_periods=10).min()
        cols[a] = mx / mn.abs().replace(0, np.nan)
    return pd.DataFrame(cols, index=panel.index)


def cand_range_cv_20d(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        hl = _ohlcv(a).reindex(s.index)
        rng = (hl["high"] - hl["low"]) / s
        m = rng.rolling(20, min_periods=12).mean()
        sd = rng.rolling(20, min_periods=12).std()
        cols[a] = sd / m.replace(0, np.nan)
    return pd.DataFrame(cols, index=panel.index)


if __name__ == "__main__":
    cands = {
        "eff_ratio_20d_skip5": cand_eff_ratio_20d_skip5,
        "candle_body_20d": cand_candle_body_20d,
        "close_pos_20d": cand_close_pos_20d,
        "vol_zscore_20x120": cand_vol_zscore_20x120,
        "kurt_20d_skip5": cand_kurt_20d_skip5,
        "shadow_ratio_20d": cand_shadow_ratio_20d,
        "maxmin_20d": cand_maxmin_20d,
        "range_cv_20d": cand_range_cv_20d,
    }
    panel = load_panel()
    macro = load_macro()
    print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets\n")
    for name, fn in cands.items():
        try:
            r = validate_factor(name, fn, direction_override=1.0)
            print(f"  GATE: pass={r['admission_gate']['pass']} "
                  f"ic={r['ic_h10']:+.4f} icir={r['icir_h10']:+.4f}\n")
        except Exception as e:
            print(f"  {name} ERROR: {e}\n")
