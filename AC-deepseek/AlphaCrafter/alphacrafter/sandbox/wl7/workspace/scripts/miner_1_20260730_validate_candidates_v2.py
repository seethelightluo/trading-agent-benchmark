"""miner_1 candidate validation (v2, robust engine based on miner_2_lib).
Universe: 15 cross-asset instruments. Factor window: 2020-01-01..2026-07-15.
Admission: |IC|>=0.007 and |ICIR|>=0.084 @ h=10; pairwise rank corr < 0.5 vs library.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from miner_2_lib import load_panel, load_macro, fwd_returns, rank_ic_series, \
    turnover_10d_rank, MIN_ASSETS, FACTOR_LAST

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def evaluate(factor, panel, horizons=(1, 2, 3, 5, 10, 20)):
    """factor: date x asset DataFrame on union calendar. Returns metrics dict."""
    fw = factor.loc[:FACTOR_LAST]
    fwd = {h: fwd_returns(panel, h) for h in horizons}
    ics = {h: rank_ic_series(fw, fwd[h]) for h in horizons}
    ic10 = ics[10]
    if len(ic10) < 200:
        return None
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    out = {"direction": direction}
    for h in horizons:
        ic = ics[h] * direction
        out[f"ic_h{h}"] = float(ic.mean())
        out[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        out[f"hit_h{h}"] = float((ic > 0).mean())
        out[f"n_h{h}"] = len(ic)
    out["coverage_asset_days"] = float(fw.notna().mean().mean())
    out["coverage_dates_ge8"] = float((fw.notna().sum(axis=1) >= MIN_ASSETS).mean())
    out["turnover_10d_rank"] = turnover_10d_rank(fw)
    return out


def pairwise_rho(a, b):
    """Mean per-date cross-sectional Spearman corr (>=8 common assets)."""
    cs = []
    common = a.index.intersection(b.index)
    for dt in common:
        if dt > pd.Timestamp(FACTOR_LAST):
            continue
        f, g = a.loc[dt], b.loc[dt]
        if isinstance(f, pd.DataFrame) or isinstance(g, pd.DataFrame):
            continue
        m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
        if int(m.sum()) >= MIN_ASSETS:
            cs.append(spearmanr(f[m].astype(float), g[m].astype(float))[0])
    return float(np.mean(cs)) if cs else float("nan")


def main():
    panel = load_panel()
    rets = panel.pct_change()
    macro = load_macro()
    v = rets.rolling(20).std()
    print(f"panel {panel.index[0].date()} .. {panel.index[-1].date()}, assets={panel.shape[1]}")

    cands = {}

    # ---- existing / recovery factors ----
    cands["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
    cands["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
    cands["vol_of_vol20x60"] = v.rolling(60).std()
    try:
        vix = macro["VIX"]
        vixr = vix.pct_change()
        beta_vix = rets.rolling(60).cov(vixr) / vixr.rolling(60).var()
        cands["vix_beta_cond_60x20"] = -beta_vix * (vix / vix.shift(20) - 1.0)
    except Exception as e:
        print("vix factor failed:", e)
    mkt = rets.mean(axis=1)
    cands["beta_ew_60d"] = rets.rolling(60).cov(mkt) / mkt.rolling(60).var().replace(0, np.nan)
    mom20 = panel.shift(5) / panel.shift(25) - 1.0
    cands["rel_mom_20d_skip5"] = mom20.sub(mom20.median(axis=1), axis=0)
    cands["max_ret_20d"] = rets.rolling(20).max()
    tot20 = rets.rolling(20).std()
    ddown20 = rets.clip(upper=0).rolling(20).std()
    cands["downside_vol_ratio_20"] = -(ddown20 / tot20.replace(0, np.nan))

    # ---- family A: trend quality ----
    def eff_ratio(win):
        num = (panel - panel.shift(win)).abs()
        den = rets.abs().rolling(win).sum()
        return num / den.replace(0, np.nan)
    cands["eff_ratio_20d"] = eff_ratio(20)
    cands["eff_ratio_60d"] = eff_ratio(60)
    cands["trend_strength_60x20"] = (panel / panel.shift(60) - 1.0).abs() / v
    ema_f = panel.ewm(span=10, adjust=False).mean()
    ema_s = panel.ewm(span=40, adjust=False).mean()
    cands["macd_10x40"] = (ema_f - ema_s) / panel

    # ---- family B: vol structure & asymmetry ----
    cands["vol_term_10x60"] = rets.rolling(10).std() / rets.rolling(60).std()
    up = rets.clip(lower=0).rolling(60).mean()
    dn = (-rets.clip(upper=0)).rolling(60).mean()
    cands["updown_ratio_60d"] = up / dn.replace(0, np.nan)
    cands["skew_60d"] = rets.rolling(60).skew()
    tot60 = rets.rolling(60).std()
    ddown60 = rets.clip(upper=0).rolling(60).std()
    cands["downside_vol_ratio_60"] = -(ddown60 / tot60.replace(0, np.nan))

    # ---- family C: macro beta ----
    try:
        dxy = macro["DXY"]
        dxyr = dxy.pct_change()
        bd = rets.rolling(60).cov(dxyr) / dxyr.rolling(60).var()
        cands["dxy_beta_60d"] = bd * (dxy / dxy.shift(20) - 1.0)
    except Exception as e:
        print("dxy failed:", e)
    u10 = panel["US10Y"]
    u10r = u10.pct_change()
    bu = rets.rolling(60).cov(u10r) / u10r.rolling(60).var()
    cands["us10y_beta_60d"] = bu * (u10 / u10.shift(20) - 1.0)

    # ---- family D: reversal ----
    delta = panel.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    cands["rsi_rev_14"] = 50 - rsi
    cands["zscore_rev_60d"] = -(panel - panel.rolling(60).mean()) / panel.rolling(60).std().replace(0, np.nan)

    # ---- evaluate ----
    results = {}
    print("\n=== Metrics (h=10 admission) ===")
    for fid, fv in cands.items():
        res = evaluate(fv, panel)
        if res is None:
            print(f"{fid}: INSUFFICIENT dates")
            continue
        results[fid] = res
        gate = abs(res["ic_h10"]) >= 0.007 and abs(res["icir_h10"]) >= 0.084
        print(f"{fid:24s} IC10={res['ic_h10']:+.4f} ICIR10={res['icir_h10']:+.4f} "
              f"hit={res['hit_h10']:.3f} n={res['n_h10']} dir={res['direction']:+.1f} "
              f"turn={res['turnover_10d_rank']:.2f} cov={res['coverage_asset_days']:.2f} "
              f"cov8={res['coverage_dates_ge8']:.2f} "
              f"decay20={res['ic_h20']:+.4f} -> {'PASS' if gate else 'fail'}")

    # ---- pairwise corr among all candidates ----
    print("\n=== Pairwise per-date rank corr (|rho|>=0.5 flagged) ===")
    names = list(results.keys())
    rho_mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                rho_mat.loc[a, b] = pairwise_rho(cands[a], cands[b])
                rho_mat.loc[b, a] = rho_mat.loc[a, b]
            else:
                rho_mat.loc[a, b] = 1.0
    for a in names:
        for b in names:
            if names.index(b) <= names.index(a):
                continue
            r = rho_mat.loc[a, b]
            flag = "!!" if abs(r) >= 0.5 else "  "
            print(f"{flag} {a} vs {b}: rho={r:.3f}")

    # ---- summary of conflicts for passing candidates ----
    print("\n=== Candidate conflict assessment (max |rho| vs other library factors) ===")
    for a in names:
        row = [abs(rho_mat.loc[a, b]) for b in names if b != a and np.isfinite(rho_mat.loc[a, b])]
        if row:
            print(f"{a:24s} max_pairwise_rho={max(row):.3f}")
    return results


if __name__ == "__main__":
    main()
