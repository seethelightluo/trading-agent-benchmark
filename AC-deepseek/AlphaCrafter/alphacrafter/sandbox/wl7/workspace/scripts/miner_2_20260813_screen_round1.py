"""miner_2 screen round 1 (2026-08-13): novel cross-asset factor families.
Candidates (one idea each, all cross-sectionally differentiated across the
15 tradable instruments; warm-up validation 2020-01-01..2026-07-15):
 1. skew_20d_skip5      - realized skewness of 20d returns (skip 5)
 2. range_vol_20d       - 20d mean intraday (high-low)/close range
 3. crypto_beta_60d     - 60d rolling beta of each asset to BTC returns
 4. bond_beta_60d       - 60d rolling beta of each asset to US10Y changes
 5. dxy_beta_cond_60x20 - -beta_to_DXY x DXY 20d change (USD-conditional)
 6. dd_recovery_60x20   - 20d change in distance-from-60d-high (recovery)
 7. vol_trend_20x120    - 20d realized vol / 120d realized vol
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_lib import validate_factor, load_panel, load_macro, WATCH, MAX_VISIBLE

_HL = {}

def _ohlcv(sym):
    if sym in _HL:
        return _HL[sym]
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
    _HL[sym] = df[["high", "low", "open", "volume"]].astype(float)
    return _HL[sym]


def cand_skew_20d_skip5(panel, macro):
    def f(s):
        r = s.pct_change()
        return r.shift(5).rolling(20, min_periods=12).skew()
    cols = {a: f(panel[a].dropna()) for a in panel.columns}
    return pd.DataFrame(cols, index=panel.index)


def cand_range_vol_20d(panel, macro):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        hl = _ohlcv(a).reindex(s.index)
        rng = (hl["high"] - hl["low"]) / s
        cols[a] = rng.rolling(20, min_periods=12).mean()
    return pd.DataFrame(cols, index=panel.index)


def _beta_to(panel, ref_name, window=60, minp=30):
    ref = panel[ref_name].pct_change()
    cols = {}
    for a in panel.columns:
        r = panel[a].pct_change()
        z = pd.concat([r.rename("r"), ref.rename("m")], axis=1).dropna()
        beta = (z["r"].rolling(window, min_periods=minp).cov(z["m"])
                / z["m"].rolling(window, min_periods=minp).var())
        cols[a] = beta
    return pd.DataFrame(cols, index=panel.index)


def cand_crypto_beta_60d(panel, macro):
    return _beta_to(panel, "BTC")


def cand_bond_beta_60d(panel, macro):
    return _beta_to(panel, "US10Y")


def cand_dxy_beta_cond_60x20(panel, macro):
    dxy = macro["DXY"]
    dxyr = dxy.pct_change()
    dxy20 = (dxy / dxy.shift(20) - 1.0)
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        z = pd.concat([r.rename("r"), dxyr.rename("m")], axis=1).dropna()
        beta = (z["r"].rolling(60, min_periods=30).cov(z["m"])
                / z["m"].rolling(60, min_periods=30).var())
        cond = (dxy20.reindex(s.index) * (-beta))
        cols[a] = cond
    return pd.DataFrame(cols, index=panel.index)


def cand_dd_recovery_60x20(panel, macro):
    def f(s):
        dd = s / s.rolling(60).max() - 1.0
        return dd - dd.shift(20)
    cols = {a: f(panel[a].dropna()) for a in panel.columns}
    return pd.DataFrame(cols, index=panel.index)


def cand_vol_trend_20x120(panel, macro):
    def f(s):
        r = s.pct_change()
        return r.rolling(20, min_periods=12).std() / r.rolling(120, min_periods=60).std()
    cols = {a: f(panel[a].dropna()) for a in panel.columns}
    return pd.DataFrame(cols, index=panel.index)


if __name__ == "__main__":
    cands = {
        "skew_20d_skip5": cand_skew_20d_skip5,
        "range_vol_20d": cand_range_vol_20d,
        "crypto_beta_60d": cand_crypto_beta_60d,
        "bond_beta_60d": cand_bond_beta_60d,
        "dxy_beta_cond_60x20": cand_dxy_beta_cond_60x20,
        "dd_recovery_60x20": cand_dd_recovery_60x20,
        "vol_trend_20x120": cand_vol_trend_20x120,
    }
    summary = []
    for name, fn in cands.items():
        try:
            r = validate_factor(name, fn)
            summary.append((name, r["ic_h10"], r["icir_h10"],
                            r.get("max_abs_library_correlation", float("nan")),
                            r["admission_gate"]["pass"]))
            if r["admission_gate"]["pass"]:
                # per-year robustness for passing candidates
                panel = load_panel()
                factor = fn(panel, load_macro()).loc[: "2026-07-15"]
                fwd = {h: None for h in (10,)}
                from miner_2_lib import fwd_returns, rank_ic_series
                ic10 = rank_ic_series(factor, fwd_returns(panel, 10))
                yrs = {}
                for y in range(2020, 2027):
                    sub = ic10.loc[str(y)]
                    if len(sub) > 20:
                        yrs[y] = (round(float(sub.mean()), 4),
                                  round(float(sub.mean() / sub.std()), 4), int(len(sub)))
                print(f"  PER-YEAR h10 IC for {name}: {yrs}")
        except Exception as e:
            print(f"{name}: ERROR {e}")
            summary.append((name, float("nan"), float("nan"), float("nan"), False))
    print("\n===== SCREEN SUMMARY R1 (h=10, warm-up 2020-01-01..2026-07-15) =====")
    for name, ic, icir, mc, ok in summary:
        flag = "PASS" if ok else "fail"
        print(f"{name:22s} IC={ic:+.4f} ICIR={icir:+.4f} maxcorr={mc:.3f} -> {flag}")
