"""miner_2 factor screening - 2026-09-29 cycle.

Explore a batch of NOVEL candidate factors against the CURRENT 8-factor library
(beta_ew_60d, corr_ew_60, downside_vol_ratio_20, dxy_beta_cond_60x20,
eurusd_beta_cond_60x20, kurt_20d_skip5, max_ret_20d, rel_mom_20d_skip5).

Data: 15 tradable cross-asset instruments, visible through 2026-09-28.
IC = cross-sectional Spearman rank IC per date (>=8 assets), horizon 10 primary.
Gates: |IC|>=0.007, |ICIR|>=0.084, max_abs_library_correlation < 0.5.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MAX_VISIBLE = "2026-09-28"
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


def row_pearson(X: np.ndarray, Y: np.ndarray, min_n: int = MIN_ASSETS) -> np.ndarray:
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


def rank_ic_fast(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    F = factor.rank(axis=1).values.astype(float)
    R = fwd.rank(axis=1).values.astype(float)
    return pd.Series(row_pearson(F, R), index=factor.index)


def turnover_10d_rank_fast(factor: pd.DataFrame) -> float:
    ranks = factor.rank(axis=1).values.astype(float)
    a, b = ranks[:-10], ranks[10:]
    valid = np.isfinite(a) & np.isfinite(b)
    cnt = valid.sum(axis=1)
    ok = cnt >= MIN_ASSETS
    m = np.full(len(a), np.nan)
    m[ok] = np.nansum(np.abs(a - b) * valid, axis=1)[ok] / cnt[ok]
    return float(np.nanmean(m))


# ---------------- library signals (current 8 effective factors) ----------------
def library_signals(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rets = panel.pct_change()
    mkt = panel.mean(axis=1).pct_change()
    out = {}
    # max_ret_20d
    out["max_ret_20d"] = rets.rolling(20, min_periods=10).max()
    # downside_vol_ratio_20 (persisted direction +1: -(dd/v))
    v = rets.rolling(20, min_periods=10).std()
    dd = rets.clip(upper=0).rolling(20, min_periods=10).std()
    out["downside_vol_ratio_20"] = -(dd / (v + EPS))
    # rel_mom_20d_skip5 (cross-sectionally demeaned)
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    out["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    # beta_ew_60d
    cov = rets.rolling(60, min_periods=30).cov(mkt)
    var = mkt.rolling(60, min_periods=30).var().replace(0, np.nan)
    out["beta_ew_60d"] = cov.div(var.to_numpy(), axis=0)
    # eurusd_beta_cond_60x20
    macro = load_macro()
    eur = macro["EURUSD"].pct_change()
    eur_var = eur.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    eb = rets.rolling(60, min_periods=30).cov(eur).div(eur_var.to_numpy(), axis=0)
    eur_mom = (macro["EURUSD"] / macro["EURUSD"].shift(20) - 1.0).reindex(panel.index)
    out["eurusd_beta_cond_60x20"] = eb.mul(eur_mom.to_numpy(), axis=0)
    # dxy_beta_cond_60x20
    dxy = macro["DXY"].pct_change()
    dxy_var = dxy.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    db = rets.rolling(60, min_periods=30).cov(dxy).div(dxy_var.to_numpy(), axis=0)
    dxy_mom = (macro["DXY"] / macro["DXY"].shift(20) - 1.0).reindex(panel.index)
    out["dxy_beta_cond_60x20"] = -db.mul(dxy_mom.to_numpy(), axis=0)
    # kurt_20d_skip5: kurtosis of returns shifted by 5 (window t-25..t-5)
    out["kurt_20d_skip5"] = rets.shift(5).rolling(20, min_periods=12).kurt()
    # corr_ew_60: mean pairwise 60d rolling correlation
    out["corr_ew_60"] = mean_pairwise_corr(rets, 60, 30)
    return out


def mean_pairwise_corr(rets: pd.DataFrame, win: int, minp: int) -> pd.DataFrame:
    n = rets.shape[1]
    cols = rets.columns
    res = {}
    for i in range(n):
        vals = []
        for j in range(n):
            if i == j:
                continue
            a, b = rets[cols[i]], rets[cols[j]]
            mab = (a * b).rolling(win, min_periods=minp).mean()
            ma = a.rolling(win, min_periods=minp).mean()
            mb = b.rolling(win, min_periods=minp).mean()
            sa = a.rolling(win, min_periods=minp).std()
            sb = b.rolling(win, min_periods=minp).std()
            c = (mab - ma * mb) / (sa * sb + EPS)
            vals.append(c)
        res[cols[i]] = pd.concat(vals, axis=1).mean(axis=1)
    return pd.DataFrame(res, index=rets.index)


# ---------------- candidate factors ----------------
def candidate_factors(panel: pd.DataFrame, ohlc: dict[str, pd.DataFrame],
                      macro: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    rets = panel.pct_change()
    out = {}
    # 1. skew_20d_skip5: realized skewness 20d, skip 5 (sibling of kurtosis)
    out["skew_20d_skip5"] = rets.shift(5).rolling(20, min_periods=12).skew()
    # 2. skew_60d_skip5: longer-window skewness
    out["skew_60d_skip5"] = rets.shift(5).rolling(60, min_periods=30).skew()
    # 3. park_vol_20d: Parkinson range vol (high-low based)
    hl = pd.DataFrame({s: np.log(ohlc[s]["high"] / ohlc[s]["low"]) for s in WATCH},
                      index=panel.index)
    park = hl.pow(2).rolling(20, min_periods=10).mean() / (4.0 * np.log(2.0))
    out["park_vol_20d"] = np.sqrt(park)
    # 4. rsi_14: Wilder RSI(14) on closes
    rsi = pd.DataFrame(index=panel.index)
    for s in WATCH:
        c = panel[s].dropna()
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi[s] = 100.0 - 100.0 / (1.0 + up / (dn + EPS))
    out["rsi_14"] = rsi
    # 5. bb_z_20d: Bollinger z-score (close - MA20)/std20
    ma20 = panel.rolling(20, min_periods=10).mean()
    sd20 = panel.rolling(20, min_periods=10).std()
    out["bb_z_20d"] = (panel - ma20) / (sd20 + EPS)
    # 6. range_pos_20d: (close - min low 20d) / (max high 20d - min low 20d)
    hi = pd.DataFrame({s: ohlc[s]["high"] for s in WATCH}, index=panel.index)
    lo = pd.DataFrame({s: ohlc[s]["low"] for s in WATCH}, index=panel.index)
    hh = hi.rolling(20, min_periods=10).max()
    ll = lo.rolling(20, min_periods=10).min()
    out["range_pos_20d"] = (panel - ll) / (hh - ll + EPS)
    # 7. vol_term_10x60: realized vol term structure vol10/vol60
    v10 = rets.rolling(10, min_periods=6).std()
    v60 = rets.rolling(60, min_periods=30).std()
    out["vol_term_10x60"] = v10 / (v60 + EPS)
    # 8. gap_mom_20d_skip5: momentum of overnight gap returns (open/prev_close - 1)
    opens = pd.DataFrame({s: ohlc[s]["open"] for s in WATCH}, index=panel.index)
    gap = opens / panel.shift(1) - 1.0
    out["gap_mom_20d_skip5"] = gap.shift(5).rolling(20, min_periods=10).mean()
    # 9. intraday_mom_20d_skip5: momentum of intraday returns (close/open - 1)
    idr = panel / opens - 1.0
    out["intraday_mom_20d_skip5"] = idr.shift(5).rolling(20, min_periods=10).mean()
    # 10. wti_beta_cond_60x20: beta vs WTI conditioned on WTI 20d momentum
    wti = panel["WTI"].pct_change()
    wv = wti.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    wb = rets.rolling(60, min_periods=30).cov(wti).div(wv.to_numpy(), axis=0)
    wm = (panel["WTI"] / panel["WTI"].shift(20) - 1.0).reindex(panel.index)
    out["wti_beta_cond_60x20"] = wb.mul(wm.to_numpy(), axis=0)
    # 11. xau_beta_cond_60x20: beta vs XAU conditioned on XAU 20d momentum
    xau = panel["XAU"].pct_change()
    xv = xau.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    xb = rets.rolling(60, min_periods=30).cov(xau).div(xv.to_numpy(), axis=0)
    xm = (panel["XAU"] / panel["XAU"].shift(20) - 1.0).reindex(panel.index)
    out["xau_beta_cond_60x20"] = xb.mul(xm.to_numpy(), axis=0)
    # 12. copper_beta_cond_60x20: beta vs COPPER conditioned on COPPER 20d momentum
    cop = panel["COPPER"].pct_change()
    cv = cop.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    cb = rets.rolling(60, min_periods=30).cov(cop).div(cv.to_numpy(), axis=0)
    cm = (panel["COPPER"] / panel["COPPER"].shift(20) - 1.0).reindex(panel.index)
    out["copper_beta_cond_60x20"] = cb.mul(cm.to_numpy(), axis=0)
    # 13. hilo_range_20d: mean((high-low)/close) over 20d (range amplitude)
    amp = pd.DataFrame({s: (ohlc[s]["high"] - ohlc[s]["low"]) / ohlc[s]["close"]
                        for s in WATCH}, index=panel.index)
    out["hilo_range_20d"] = amp.rolling(20, min_periods=10).mean()
    # 14. updown_asym_20d: mean(pos ret)/|mean(neg ret)| over 20d (skew proxy)
    up = rets.where(rets > 0, 0.0).rolling(20, min_periods=10).mean()
    dn = rets.where(rets < 0, 0.0).rolling(20, min_periods=10).mean()
    out["updown_asym_20d"] = up / (dn.abs() + EPS)
    return out


def validate_candidate(name: str, factor: pd.DataFrame, panel: pd.DataFrame,
                       fwd: dict[int, pd.DataFrame], libs: dict[str, pd.DataFrame],
                       year_split: bool = True) -> dict:
    factor = factor.reindex(panel.index)
    n_valid = int(factor.notna().sum().sum())
    if n_valid < 200:
        return {"name": name, "admission_gate": {"pass": False}, "reason": "insufficient_data",
                "n_valid": n_valid}
    res = {"name": name, "n_assets": panel.shape[1], "n_rows": len(factor)}
    ic_by_h = {}
    for h in HORIZONS:
        F = factor.rank(axis=1).values.astype(float)
        R = fwd[h].rank(axis=1).values.astype(float)
        ic_by_h[h] = pd.Series(row_pearson(F, R), index=factor.index)
    ic10 = ic_by_h[10].dropna()
    direction = float(np.sign(ic10.mean())) if len(ic10) and ic10.mean() != 0 else 1.0
    for h in HORIZONS:
        ic = ic_by_h[h].dropna() * direction
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean())
        res[f"n_dates_h{h}"] = int(len(ic))
    res["direction"] = direction
    valid = factor.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank_fast(factor)
    # library correlation over last 700 overlapping dates
    idx = factor.index[-700:]
    F = factor.loc[idx].rank(axis=1).values.astype(float)
    per = {}
    for fid, lf in libs.items():
        L = lf.reindex(idx).rank(axis=1).values.astype(float)
        rs = row_pearson(F, L)
        per[fid] = round(float(np.nanmean(rs)), 4)
    valid_c = [abs(v) for v in per.values() if np.isfinite(v)]
    res["max_abs_library_correlation"] = round(max(valid_c), 4) if valid_c else float("nan")
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in HORIZONS}
    gate_ic = abs(res["ic_h10"]) >= ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= ADMISSION["icir"]
    gate_corr = (not np.isfinite(res["max_abs_library_correlation"])
                 or res["max_abs_library_correlation"] < ADMISSION["corr"])
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "corr_pass": bool(gate_corr),
                             "pass": bool(gate_ic and gate_icir and gate_corr)}
    if year_split:
        ic = ic_by_h[10] * direction
        yr = {}
        for y, grp in ic.groupby(ic.index.year):
            grp = grp.dropna()
            if len(grp) > 30:
                yr[str(y)] = {"ic": round(float(grp.mean()), 4),
                             "icir": round(float(grp.mean() / grp.std()), 4) if grp.std() > 0 else None,
                             "n": int(len(grp))}
        res["yearly_ic_h10"] = yr
        # recent 12-month window
        recent = ic[ic.index >= "2025-09-01"].dropna()
        if len(recent) > 30:
            res["recent_12m"] = {"ic": round(float(recent.mean()), 4),
                                 "icir": round(float(recent.mean() / recent.std()), 4),
                                 "n": int(len(recent))}
    flag = "PASS" if res["admission_gate"]["pass"] else "FAIL"
    print(f"  {name:<26} h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} "
          f"hit={res['hit_h10']:.3f} cov={res['coverage_asset_days']:.3f} "
          f"turn={res['turnover_10d_rank']:.2f} maxcorr={res['max_abs_library_correlation']} "
          f"-> {flag}", flush=True)
    return res


def main():
    print("Loading panel through", MAX_VISIBLE)
    panel = load_panel()
    print("Panel shape:", panel.shape)
    ohlc = load_ohlc()
    macro = load_macro()
    fwd = {h: fwd_returns(panel, h) for h in HORIZONS}

    print("\nBuilding current library signals (8 factors)...")
    libs = library_signals(panel)
    for fid, lf in libs.items():
        print(f"  lib {fid:<28} rows={len(lf)} valid={int(lf.notna().sum().sum())}")

    print("\nBuilding candidate factors...")
    cands = candidate_factors(panel, ohlc, macro)
    print("Candidates:", list(cands.keys()))

    print("\n=== VALIDATION (gates: |IC|>=0.007, |ICIR|>=0.084, corr<0.5 @ h10) ===")
    results = {}
    for name, factor in cands.items():
        results[name] = validate_candidate(name, factor, panel, fwd, libs)
    print("\n=== PASSING CANDIDATES ===")
    for name, r in results.items():
        if r.get("admission_gate", {}).get("pass"):
            print(f"  {name}: IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
                  f"maxcorr={r['max_abs_library_correlation']} "
                  f"recent12m={r.get('recent_12m')}")
            print("    yearly:", json.dumps(r.get("yearly_ic_h10", {})))
    out = {"date": MAX_VISIBLE, "results": results}
    with open("scripts/miner_2_20260929_screen_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nSaved results to scripts/miner_2_20260929_screen_results.json")


if __name__ == "__main__":
    main()
