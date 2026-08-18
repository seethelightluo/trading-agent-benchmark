"""miner_2 factor screening - 2027-03-16 cycle (vectorized).

Screen NOVEL candidate factors against the CURRENT 7-factor active library.
IC = cross-sectional Spearman rank IC per date (>=8 assets), horizon 10 primary.
Gates (benchmark contract): |IC|>=0.007, |ICIR|>=0.084 @ h=10.
Data visible through 2027-03-15 (current date 2027-03-16).
Also reports recent-window (last 250d) IC/ICIR for freshness.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MAX_VISIBLE = "2027-03-15"
MIN_ASSETS = 8
ADMISSION = {"ic": 0.007, "icir": 0.084}
HORIZONS = (1, 2, 3, 5, 10, 20)
EPS = 1e-12

# active library (from factor_ensemble.json, 2027-03-02 cycle)
ACTIVE_LIB = ["rel_mom_20d_skip5", "beta_ew_60d", "downside_vol_ratio_20",
              "corr_ew_60", "eurusd_beta_cond_60x20", "max_ret_20d", "kurt_20d_skip5"]


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
    F = factor.loc[common]
    R = fwd.loc[common]
    Fr = F.rank(axis=1)
    Rr = R.rank(axis=1)
    ics = row_pearson(Fr.values, Rr.values)
    return pd.Series(ics, index=common).sort_index()


def turnover_10d_rank(factor: pd.DataFrame) -> float:
    ranks = factor.rank(axis=1)
    out = []
    for i in range(10, len(ranks)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) < MIN_ASSETS:
            continue
        out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


def _beta(asset_ret: pd.Series, ref_ret: pd.Series, w=60, minp=30) -> pd.Series:
    z = pd.concat([asset_ret.rename("r"), ref_ret.rename("m")], axis=1).dropna()
    return z["r"].rolling(w, min_periods=minp).cov(z["m"]) / z["m"].rolling(w, min_periods=minp).var()


# ---------------- library factors (7 active) ----------------
def library_signals(panel: pd.DataFrame, macro: dict) -> dict[str, pd.DataFrame]:
    out = {}
    rets = panel.pct_change()
    ew = rets.mean(axis=1)
    mom20 = panel.shift(5) / panel.shift(25) - 1.0
    out["rel_mom_20d_skip5"] = mom20.sub(mom20.median(axis=1), axis=0)
    cov = rets.rolling(60).cov(ew)
    var = ew.rolling(60).var()
    out["beta_ew_60d"] = cov.div(var, axis=0)
    neg = rets.where(rets < 0, 0.0)
    dv = -((neg ** 2).rolling(20).mean() ** 0.5) / rets.rolling(20).std()
    out["downside_vol_ratio_20"] = dv
    out["corr_ew_60"] = rets.rolling(60).corr(ew)
    vixr = macro["VIX"].pct_change()
    vix20 = macro["VIX"] / macro["VIX"].shift(20) - 1.0
    out["eurusd_beta_cond_60x20"] = pd.DataFrame(
        {a: _beta(rets[a], macro["EURUSD"].pct_change()) * (macro["EURUSD"] / macro["EURUSD"].shift(20) - 1.0).reindex(rets[a].dropna().index)
         for a in panel.columns}, index=panel.index)
    out["max_ret_20d"] = rets.rolling(20).max()
    out["kurt_20d_skip5"] = rets.shift(5).rolling(20).kurt()
    return out


# ---------------- candidate factors ----------------
def make_candidates(panel: pd.DataFrame, ohlc: dict, macro: dict) -> dict[str, pd.DataFrame]:
    rets = panel.pct_change()
    ew = rets.mean(axis=1)
    C = {}

    # 1. RSI(14) cross-sectionally demeaned (mean-reversion pressure)
    def _rsi(s, w=14):
        d = s.diff()
        up = d.clip(lower=0).rolling(w).mean()
        dn = (-d.clip(upper=0)).rolling(w).mean()
        rs = up / (dn + EPS)
        return 100 - 100 / (1 + rs)
    rsi = pd.DataFrame({a: _rsi(panel[a]) for a in panel.columns}, index=panel.index)
    C["rsi14_dm"] = rsi.sub(rsi.median(axis=1), axis=0)

    # 2. 20d range position: (close - low20)/(high20 - low20) - 0.5
    hi = panel.rolling(20).max()
    lo = panel.rolling(20).min()
    C["range_pos_20"] = (panel - lo) / (hi - lo + EPS) - 0.5

    # 3. upside/downside vol ratio 20d (up-day vol vs down-day vol)
    up = rets.where(rets > 0, np.nan)
    dn = rets.where(rets < 0, np.nan)
    upvol = up.rolling(20).std()
    dnvol = dn.rolling(20).std()
    C["vol_updown_ratio_20"] = upvol / (dnvol + EPS)

    # 4. skewness 20d skip5
    C["skew_20d_skip5"] = rets.shift(5).rolling(20).skew()

    # 5. drawdown from 60d high (negative distance)
    C["drawdown_60d"] = panel / panel.rolling(60).max() - 1.0

    # 6. Bollinger bandwidth 20x2 (squeeze)
    ma = panel.rolling(20).mean()
    sd = rets.rolling(20).std() * panel
    C["boll_bw_20"] = (4 * sd) / (ma + EPS)

    # 7. MACD normalized: (EMA12 - EMA26)/close
    e12 = panel.ewm(span=12, adjust=False).mean()
    e26 = panel.ewm(span=26, adjust=False).mean()
    C["macd_norm"] = (e12 - e26) / (panel + EPS)

    # 8. dual-MA trend: close/MA20 - close/MA60
    ma20 = panel.rolling(20).mean()
    ma60 = panel.rolling(60).mean()
    C["dual_ma_spread"] = panel / (ma20 + EPS) - panel / (ma60 + EPS)

    # 9. upper/lower shadow ratio 20d (from OHLC)
    sh = {}
    for a in panel.columns:
        o = ohlc[a]["open"]
        h = ohlc[a]["high"]
        l = ohlc[a]["low"]
        c = ohlc[a]["close"]
        upper = (h - np.maximum(o, c)).rolling(20).mean()
        lower = (np.minimum(o, c) - l).rolling(20).mean()
        sh[a] = upper / (lower + EPS)
    C["shadow_ratio_20"] = pd.DataFrame(sh, index=panel.index)

    # 10. US10Y beta conditional on bond 20d trend (bond-beta conditional)
    us10 = panel["US10Y"].pct_change()
    us10_20 = panel["US10Y"] / panel["US10Y"].shift(20) - 1.0
    C["us10y_beta_cond_60x20"] = pd.DataFrame(
        {a: _beta(rets[a], us10) * us10_20.reindex(rets[a].dropna().index)
         for a in panel.columns}, index=panel.index)

    # 11. momentum acceleration: mom20_skip5 - mom10_skip3
    mom20 = panel.shift(5) / panel.shift(25) - 1.0
    mom10 = panel.shift(3) / panel.shift(13) - 1.0
    C["mom_accel_20x10"] = mom20 - mom10

    # 12. downside beta to EW (beta on down-market days only)
    db = {}
    for a in panel.columns:
        r = rets[a]
        z = pd.concat([r.rename("r"), ew.rename("m")], axis=1).dropna()
        mneg = z["m"] < 0
        if mneg.sum() > 30:
            seg = z[mneg]
            db[a] = seg["r"].rolling(60, min_periods=20).cov(seg["m"]) / seg["m"].rolling(60, min_periods=20).var()
        else:
            db[a] = pd.Series(np.nan, index=z.index)
    C["down_beta_ew_60"] = pd.DataFrame(db, index=panel.index)

    # 13. vol ratio 10x60 (short-term vol vs long-term vol regime)
    C["vol_ratio_10x60"] = rets.rolling(10).std() / (rets.rolling(60).std() + EPS)

    # 14. volume expansion: 20d mean volume / 60d mean volume
    ve = {}
    for a in panel.columns:
        v = ohlc[a]["volume"]
        ve[a] = v.rolling(20).mean() / (v.rolling(60).mean() + EPS)
    C["vol_exp_20x60"] = pd.DataFrame(ve, index=panel.index)

    # 15. price/volume correlation 20d (volume-confirmed moves)
    pv = {}
    for a in panel.columns:
        r = rets[a].rename("r")
        v = ohlc[a]["volume"].rename("v")
        z = pd.concat([r, v], axis=1).dropna()
        pv[a] = z["r"].rolling(20).corr(z["v"])
    C["pv_corr_20"] = pd.DataFrame(pv, index=panel.index)

    # 16. XAU risk-off conditional beta (gold-beta conditional on gold 20d trend)
    xau_ret = panel["XAU"].pct_change()
    xau_20 = panel["XAU"] / panel["XAU"].shift(20) - 1.0
    C["xau_beta_cond_60x20"] = pd.DataFrame(
        {a: _beta(rets[a], xau_ret) * xau_20.reindex(rets[a].dropna().index)
         for a in panel.columns}, index=panel.index)

    return C


def datewise_corr(factor: pd.DataFrame, lf: pd.DataFrame, n_dates=700) -> float:
    cs = []
    common = factor.index.intersection(lf.index)[-n_dates:]
    for dt in common:
        f = factor.loc[dt]
        g = lf.loc[dt]
        if isinstance(f, pd.DataFrame) or isinstance(g, pd.DataFrame):
            continue
        m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
        m = m.reindex(f.index).fillna(False)
        if int(m.sum()) >= MIN_ASSETS:
            cs.append(pd.Series(f[m]).corr(pd.Series(g[m]), method="spearman"))
    return float(np.mean(cs)) if cs else float("nan")


def main():
    panel = load_panel()
    ohlc = load_ohlc()
    macro = load_macro()
    rets = panel.pct_change()
    print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets, last={panel.index[-1].date()}")

    libs = library_signals(panel, macro)
    cands = make_candidates(panel, ohlc, macro)
    fwd = {h: fwd_returns(panel, h) for h in HORIZONS}

    # warm-up window used for admission (full visible history; robust across regimes)
    admit_end = panel.index[-1]
    admit_start = panel.index[0]

    rows = []
    for name, fac in cands.items():
        fw = fac.loc[admit_start:admit_end]
        ic10 = rank_ic_series(fw, fwd[10])
        if len(ic10) < 50:
            rows.append({"name": name, "error": "too few IC dates"})
            continue
        direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
        row = {"name": name, "direction": direction}
        ic_adj = {}
        for h in HORIZONS:
            ic = rank_ic_series(fw, fwd[h]) * direction
            ic_adj[h] = ic
            icir = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
            row[f"ic_h{h}"] = float(ic.mean())
            row[f"icir_h{h}"] = icir
            row[f"hit_h{h}"] = float((ic > 0).mean()) if len(ic) else float("nan")
            if h == 10:
                row["n_dates"] = int(len(ic))
                row["raw_ic_h10"] = float(rank_ic_series(fw, fwd[10]).mean())
        # freshness: last 250 IC dates at h=10
        ic_recent = ic_adj[10].tail(250)
        row["recent_ic_h10"] = float(ic_recent.mean())
        row["recent_icir_h10"] = float(ic_recent.mean() / ic_recent.std()) if len(ic_recent) > 2 and ic_recent.std() > 0 else float("nan")
        valid = fw.notna()
        row["coverage_asset_days"] = float(valid.mean().mean())
        row["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
        row["turnover_10d_rank"] = turnover_10d_rank(fw)
        # library correlation
        max_corr, per = 0.0, {}
        for lid, lf in libs.items():
            c = datewise_corr(fw, lf)
            per[lid] = c
            if c is not None and np.isfinite(c):
                max_corr = max(max_corr, abs(c))
        row["max_abs_library_correlation"] = round(max_corr, 4)
        row["lib_corr"] = {k: (round(v, 3) if v is not None and np.isfinite(v) else None) for k, v in per.items()}
        row["pass_ic"] = abs(row["ic_h10"]) >= ADMISSION["ic"]
        row["pass_icir"] = abs(row["icir_h10"]) >= ADMISSION["icir"]
        row["PASS"] = row["pass_ic"] and row["pass_icir"]
        rows.append(row)

    res = pd.DataFrame(rows).sort_values("ic_h10", key=lambda s: s.abs(), ascending=False)
    cols = ["name", "direction", "ic_h10", "icir_h10", "hit_h10", "n_dates",
            "recent_ic_h10", "recent_icir_h10", "coverage_asset_days", "coverage_dates_ge8",
            "turnover_10d_rank", "max_abs_library_correlation", "PASS"]
    print("\n=== CANDIDATE SUMMARY (h=10, full history thru 2027-03-15) ===")
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== DECAY (IC by horizon, direction-adjusted) ===")
    for _, r in res.iterrows():
        print(f"{r['name']:<26} " + " ".join(f"h{h}:{r[f'ic_h{h}']:.4f}" for h in HORIZONS))

    print("\n=== LIB CORR vs ACTIVE LIBRARY ===")
    for _, r in res.iterrows():
        print(f"{r['name']:<26} max={r.get('max_abs_library_correlation')} " +
              " ".join(f"{k}:{v}" for k, v in r.get("lib_corr", {}).items()))

    passing = res[res["PASS"]]
    print(f"\nPASSING: {len(passing)}")
    for _, r in passing.iterrows():
        print(f"  {r['name']}: ic={r['ic_h10']:.4f} icir={r['icir_h10']:.4f} "
              f"recent_ic={r['recent_ic_h10']:.4f} max_lib_corr={r['max_abs_library_correlation']} "
              f"turn={r['turnover_10d_rank']:.2f} cov={r['coverage_dates_ge8']:.2f}")


if __name__ == "__main__":
    main()
