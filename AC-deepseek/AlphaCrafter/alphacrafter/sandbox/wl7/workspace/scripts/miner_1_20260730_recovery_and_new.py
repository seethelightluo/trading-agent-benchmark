"""
miner_1: recovery + new-candidate validation on the 15-asset cross-asset universe.
Window: factor dates 2020-01-01..2026-07-15 (data visible through 2026-07-29).
Admission gates (benchmark-wide): |IC|>=0.007, |ICIR|>=0.084 at horizon 10.
Also computes pairwise per-date rank correlations among candidates (gate rho<0.5).
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner_1_metrics import (load_panel, panel_col, evaluate, gate_pass,
                             library_corr, library_signal)

def main():
    frames = load_panel()
    closes = panel_col(frames, "close")
    opens = panel_col(frames, "open")
    highs = panel_col(frames, "high")
    lows = panel_col(frames, "low")
    rets = closes.pct_change()
    print(f"panel: {closes.index[0].date()} .. {closes.index[-1].date()}, assets={closes.shape[1]}")

    cands = {}

    # ---------- recovery: previously effective core factors ----------
    cands["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
    cands["mom_60d_skip5"] = closes.shift(5) / closes.shift(65) - 1.0
    cands["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
    v = rets.rolling(20).std()
    cands["vol_of_vol20x60"] = v.rolling(60).std()
    # VIX-beta conditional (VIX is observation-only macro)
    try:
        vix = load_macro("VIX")["close"].astype(float)
        vixr = vix.pct_change()
        beta_vix = rets.rolling(60).cov(vixr) / vixr.rolling(60).var()
        cands["vix_beta_cond_60x20"] = -beta_vix * (vix / vix.shift(20) - 1.0)
    except Exception as e:
        print("vix factor failed:", e)

    # ---------- new family A: trend quality / efficiency ----------
    def eff_ratio(win):
        num = (closes - closes.shift(win)).abs()
        den = rets.abs().rolling(win).sum()
        return num / den.replace(0, np.nan)
    cands["eff_ratio_20d"] = eff_ratio(20)
    cands["eff_ratio_60d"] = eff_ratio(60)
    # trend strength: |mom| / vol
    cands["trend_strength_60x20"] = (closes / closes.shift(60) - 1.0).abs() / v
    # MACD-type: EMA(short) - EMA(long)
    ema_f = closes.ewm(span=10, adjust=False).mean()
    ema_s = closes.ewm(span=40, adjust=False).mean()
    cands["macd_10x40"] = (ema_f - ema_s) / closes

    # ---------- new family B: vol term structure & asymmetry ----------
    cands["vol_term_10x60"] = rets.rolling(10).std() / rets.rolling(60).std()
    up = rets.clip(lower=0).rolling(60).mean()
    dn = (-rets.clip(upper=0)).rolling(60).mean()
    cands["updown_ratio_60d"] = up / dn.replace(0, np.nan)
    cands["skew_60d"] = rets.rolling(60).skew()
    # downside semi-vol ratio (flip: high = defensive)
    dvol = rets.clip(upper=0).rolling(60).std()
    cands["downside_vol_ratio_60"] = -dvol / v

    # ---------- new family C: macro beta (DXY / US10Y) ----------
    try:
        dxy = load_macro("DXY")["close"].astype(float)
        dxyr = dxy.pct_change()
        bd = rets.rolling(60).cov(dxyr) / dxyr.rolling(60).var()
        cands["dxy_beta_60d"] = bd * (dxy / dxy.shift(20) - 1.0)
    except Exception as e:
        print("dxy factor failed:", e)
    try:
        u10 = closes["US10Y"]
        u10r = u10.pct_change()
        bu = rets.rolling(60).cov(u10r) / u10r.rolling(60).var()
        cands["us10y_beta_60d"] = bu * (u10 / u10.shift(20) - 1.0)
    except Exception as e:
        print("us10y factor failed:", e)

    # ---------- new family D: RSI / short-term reversal ----------
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    cands["rsi_rev_14"] = 50 - rsi  # contrarian: low RSI -> high signal
    cands["zscore_rev_60d"] = -(closes - closes.rolling(60).mean()) / closes.rolling(60).std()

    # ---------- evaluate all ----------
    lib_ids = ["mom_10d_skip5", "mom_120d_skip5", "vol_of_vol20x60", "vix_beta_cond_60x20"]
    results = []
    for fid, fv in cands.items():
        res = evaluate(fv, closes, horizon=10, label=fid)
        if res is None:
            print(f"{fid}: FAILED (insufficient dates)")
            continue
        mx, per = library_corr(fv, closes, library_ids=lib_ids)
        res["max_abs_library_correlation"] = mx
        res["gate"] = "PASS" if gate_pass(res) else "FAIL"
        results.append(res)
        print(f"{fid}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} hit={res['ic_hit_ratio']:.3f} "
              f"n={res['n_ic_dates']} turn={res['turnover_10d_rank']:.2f} cov={res['coverage_asset_days']:.2f} "
              f"libcorr={mx} decay10={res['decay_ic_by_horizon']['10']} [{res['gate']}]")

    # ---------- pairwise per-date rank correlation among all candidates ----------
    print("\n--- pairwise per-date rank corr (mean over common factor dates) ---")
    names = [r["factor_id"] for r in results]
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            f = cands[names[i]].rank(axis=1)
            g = cands[names[j]].rank(axis=1)
            cs = []
            for dt in f.index.intersection(g.index):
                if dt > pd.Timestamp("2026-07-15"):
                    continue
                m = f.loc[dt].notna() & g.loc[dt].notna()
                if m.sum() >= 8:
                    cs.append(f.loc[dt][m].corr(g.loc[dt][m], method="spearman"))
            rho = float(np.mean(cs)) if cs else np.nan
            pairs[f"{names[i]}|{names[j]}"] = round(rho, 3)
            print(f"{names[i]} vs {names[j]}: rho={rho:.3f}")

    return results, pairs

if __name__ == "__main__":
    main()
