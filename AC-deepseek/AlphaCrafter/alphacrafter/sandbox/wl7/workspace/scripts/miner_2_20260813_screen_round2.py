"""miner_2 screen round 2 (2026-08-13): macro/asset beta-linkage families.
Candidates (warm-up validation 2020-01-01..2026-07-15, h=10 admission):
 1. oil_beta_60d     - 60d rolling beta of each asset to WTI returns
 2. gold_beta_60d    - 60d rolling beta of each asset to XAU returns
 3. ndx_beta_60d     - 60d rolling beta of each asset to NDX returns (tech beta)
 4. jpy_beta_cond_60x20 - beta(asset, USDJPY, 60) x USDJPY 20d change
 5. drawdown_60d     - distance from 60d high (level, negative = drawdown)
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_lib import validate_factor, load_panel, load_macro, WATCH


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


def cand_oil_beta_60d(panel, macro):
    return _beta_to(panel, "WTI")


def cand_gold_beta_60d(panel, macro):
    return _beta_to(panel, "XAU")


def cand_ndx_beta_60d(panel, macro):
    return _beta_to(panel, "NDX")


def cand_jpy_beta_cond_60x20(panel, macro):
    jpy = macro["USDJPY"]
    jpyr = jpy.pct_change()
    jpy20 = (jpy / jpy.shift(20) - 1.0)
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        r = s.pct_change()
        z = pd.concat([r.rename("r"), jpyr.rename("m")], axis=1).dropna()
        beta = (z["r"].rolling(60, min_periods=30).cov(z["m"])
                / z["m"].rolling(60, min_periods=30).var())
        cols[a] = beta * jpy20.reindex(s.index)
    return pd.DataFrame(cols, index=panel.index)


def cand_drawdown_60d(panel, macro):
    def f(s):
        return s / s.rolling(60).max() - 1.0
    cols = {a: f(panel[a].dropna()) for a in panel.columns}
    return pd.DataFrame(cols, index=panel.index)


if __name__ == "__main__":
    cands = {
        "oil_beta_60d": cand_oil_beta_60d,
        "gold_beta_60d": cand_gold_beta_60d,
        "ndx_beta_60d": cand_ndx_beta_60d,
        "jpy_beta_cond_60x20": cand_jpy_beta_cond_60x20,
        "drawdown_60d": cand_drawdown_60d,
    }
    summary = []
    for name, fn in cands.items():
        try:
            r = validate_factor(name, fn)
            summary.append((name, r["ic_h10"], r["icir_h10"],
                            r.get("max_abs_library_correlation", float("nan")),
                            r["admission_gate"]["pass"]))
            if r["admission_gate"]["pass"]:
                from miner_2_lib import fwd_returns, rank_ic_series
                panel = load_panel()
                factor = fn(panel, load_macro()).loc[: "2026-07-15"]
                ic10 = rank_ic_series(factor, fwd_returns(panel, 10)) * r["direction"]
                yrs = {}
                for y in range(2020, 2027):
                    sub = ic10.loc[str(y)]
                    if len(sub) > 20:
                        yrs[y] = (round(float(sub.mean()), 4),
                                  round(float(sub.mean() / sub.std()), 4), int(len(sub)))
                print(f"  PER-YEAR h10 IC (dir adj) for {name}: {yrs}")
        except Exception as e:
            print(f"{name}: ERROR {e}")
            summary.append((name, float("nan"), float("nan"), float("nan"), False))
    print("\n===== SCREEN SUMMARY R2 (h=10, warm-up 2020-01-01..2026-07-15) =====")
    for name, ic, icir, mc, ok in summary:
        flag = "PASS" if ok else "fail"
        print(f"{name:22s} IC={ic:+.4f} ICIR={icir:+.4f} maxcorr={mc:.3f} -> {flag}")
