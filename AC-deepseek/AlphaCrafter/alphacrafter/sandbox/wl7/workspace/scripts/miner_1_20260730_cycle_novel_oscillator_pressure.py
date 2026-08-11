"""miner_1: cycle validation of NOVEL factor families (oscillators, buying pressure,
volume-return co-movement, trend quality, serial autocorrelation, intermediate momentum,
range-vs-close vol structure) on the 15-instrument cross-asset benchmark.
Admission gates (benchmark contract): |IC10| >= 0.007, |ICIR10| >= 0.084, library |rho| < 0.5.
Factor dates <= 2026-07-15; data visible through 2026-07-29.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from miner_1_metrics import load_panel, load_macro, WATCH, MIN_ASSETS, FACTOR_LAST, MAX_VISIBLE, evaluate, gate_pass

WATCH = ["000300.SH", "000688.SH", "SPX", "NDX", "SOX", "HSI", "N225", "SX5E",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def per_asset(fn):
    """Apply fn on each asset's own trading calendar, reindex to union panel."""
    def wrapper(panel, *series_list):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            args = tuple(sr[a].reindex(s.index) for sr in series_list)
            cols[a] = fn(s, *args)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


def library_signals(closes, vols, vix, dxy, usdjpy, eurusd):
    """Recompute all 9 currently-effective library factor signals."""
    rets = closes.pct_change()
    mkt = closes.mean(axis=1)
    mkt_r = mkt.pct_change()
    out = {}
    out["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
    out["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
    v = rets.rolling(20).std()
    out["vol_of_vol20x60"] = v.rolling(60).std()
    out["max_ret_20d"] = rets.rolling(20).max()
    out["downside_vol_ratio_20"] = -(rets.clip(upper=0).rolling(20).std() / rets.rolling(20).std())
    mom20 = closes.shift(5) / closes.shift(25) - 1.0
    out["rel_mom_20d_skip5"] = mom20.sub(mom20.median(axis=1), axis=0)
    out["beta_ew_60d"] = rets.rolling(60).cov(mkt_r) / mkt_r.rolling(60).var()
    vixr = vix.pct_change()
    out["vix_beta_cond_60x20"] = -(rets.rolling(60).cov(vixr) / vixr.rolling(60).var()) * (vix / vix.shift(20) - 1.0)
    amihud = (rets.abs() / vols.replace(0, np.nan))
    out["amihud_20"] = amihud.rolling(20).mean()
    return out


def library_corr(fvals, closes, libs, n_days=500):
    """Max abs mean per-date cross-sectional rank corr vs library factors."""
    out = {}
    common = fvals.index.intersection(closes.index)
    for fid, lf in libs.items():
        cs = []
        for dt in common[-n_days:]:
            f = fvals.loc[dt]
            g = lf.loc[dt].reindex(f.index)
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            if int(m.sum()) >= MIN_ASSETS:
                r, _ = spearmanr(f[m].astype(float), g[m].astype(float))
                cs.append(r)
        out[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in out.values() if v is not None]
    return (round(max(valid), 4) if valid else None), out


# ---------------- main ----------------
if __name__ == "__main__":
    frames = load_panel()
    closes = pd.DataFrame({s: f["close"].astype(float) for s, f in frames.items()}).sort_index()
    opens = pd.DataFrame({s: f["open"].astype(float) for s, f in frames.items()}).sort_index()
    highs = pd.DataFrame({s: f["high"].astype(float) for s, f in frames.items()}).sort_index()
    lows = pd.DataFrame({s: f["low"].astype(float) for s, f in frames.items()}).sort_index()
    vols = pd.DataFrame({s: f["volume"].astype(float) for s, f in frames.items()}).sort_index()
    macro = {m: load_macro(m)["close"].astype(float) for m in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]}
    vix, dxy, usdjpy, eurusd = macro["VIX"], macro["DXY"], macro["USDJPY"], macro["EURUSD"]
    rets = closes.pct_change()
    print(f"panel {closes.index[0].date()}..{closes.index[-1].date()} assets={closes.shape[1]} rows={len(closes)}")

    cands = {}

    # ---- Family A: oscillator / mean-reversion signatures ----
    # RSI(14): 100 - 100/(1+avg_gain/avg_loss)
    def rsi(s, n=14):
        d = s.diff()
        up = d.clip(lower=0).rolling(n).mean()
        dn = (-d.clip(upper=0)).rolling(n).mean()
        rs = up / dn.replace(0, np.nan)
        return 100.0 - 100.0 / (1.0 + rs)
    cands["rsi_14"] = per_asset(rsi)(closes)
    # Bollinger z-score 20d: (close - sma20)/std20
    cands["bbz_20d"] = per_asset(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std().replace(0, np.nan))(closes)
    # MACD normalized: (EMA12 - EMA26) / (vol 20d * close)
    def macd_norm(s):
        macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        vol = s.pct_change().rolling(20).std()
        return macd / (vol * s).replace(0, np.nan)
    cands["macd_norm_12x26"] = per_asset(macd_norm)(closes)

    # ---- Family B: buying pressure / volume-return co-movement ----
    def up_vol_share(s, v):
        r = s.pct_change()
        upv = (v * r.gt(0)).rolling(20).sum()
        totv = v.rolling(20).sum()
        return upv / totv.replace(0, np.nan) - 0.5
    cands["upvol_share_20d"] = per_asset(up_vol_share)(closes, vols)
    def vol_ret_corr(s, v):
        r = s.pct_change()
        return r.rolling(20).corr(v)
    cands["vol_ret_corr_20d"] = per_asset(vol_ret_corr)(closes, vols)

    # ---- Family C: trend quality / serial structure ----
    def trend_r2(s, win=60):
        x = np.arange(win)
        def r2(y):
            if not np.all(np.isfinite(y)) or np.std(y) == 0:
                return np.nan
            c = np.polyfit(x, y, 1)
            yhat = np.polyval(c, x)
            ss = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)
            return ss
        return s.rolling(win).apply(lambda w: r2(np.log(w)), raw=True)
    cands["trend_r2_60d"] = per_asset(trend_r2)(closes)
    def serial_ac(s):
        def ac(w):
            if len(w) < 6 or np.std(w[:-1]) == 0 or np.std(w[1:]) == 0:
                return np.nan
            return np.corrcoef(w[:-1], w[1:])[0, 1]
        return s.pct_change().rolling(20).apply(ac, raw=True)
    cands["serial_ac_20d"] = per_asset(serial_ac)(closes)

    # ---- Family D: intermediate momentum 60d (skip5), raw & cross-sectional z ----
    mom60 = per_asset(lambda s: s.shift(5) / s.shift(65) - 1.0)(closes)
    cands["mom_60d_skip5"] = mom60
    cands["z_mom_60d_skip5"] = mom60.sub(mom60.mean(axis=1), axis=0).div(mom60.std(axis=1).replace(0, np.nan), axis=0)

    # ---- Family E: range-vs-close volatility structure ----
    def parkinson_ratio(s, hi, lo):
        r = np.log(hi / lo)
        pv = (r ** 2).rolling(20).mean() / (4.0 * np.log(2.0))
        cv = s.pct_change().rolling(20).var()
        return np.sqrt(pv / cv.replace(0, np.nan))
    def pr(a):
        s = closes[a].dropna(); hi = highs[a].dropna(); lo = lows[a].dropna()
        idx = s.index.intersection(hi.index).intersection(lo.index)
        return parkinson_ratio(s.loc[idx], hi.loc[idx], lo.loc[idx])
    cands["parkinson_vs_close_20d"] = pd.DataFrame({a: pr(a) for a in closes.columns}, index=closes.index)
    cands["vol_20x120_ratio"] = rets.rolling(20).std() / rets.rolling(120).std().replace(0, np.nan)

    # ---- evaluate ----
    libs = library_signals(closes, vols, vix, dxy, usdjpy, eurusd)
    print("\n=== Metrics (h=10 admission) ===")
    results = {}
    for fid, fv in cands.items():
        res = evaluate(fv, closes, label=fid)
        if res is None:
            print(f"{fid}: INSUFFICIENT dates")
            continue
        results[fid] = res
        gate = gate_pass(res)
        print(f"{fid:24s} IC10={res['ic']:+.4f} ICIR10={res['icir']:+.4f} "
              f"hit={res['ic_hit_ratio']:.3f} n={res['n_ic_dates']} "
              f"turn={res['turnover_10d_rank']:.2f} cov={res['coverage_asset_days']:.2f} "
              f"decay20={res['decay_ic_by_horizon']['20']:+.4f} -> {'PASS' if gate else 'fail'}")

    print("\n=== Library correlation (max |rho| over last 500d) ===")
    for fid in results:
        maxrho, per = library_corr(cands[fid], closes, libs)
        results[fid]["max_abs_library_correlation"] = maxrho
        flag = "OK" if (maxrho is not None and maxrho < 0.5) else "HIGH-CORR"
        print(f"{fid:24s} max_rho={maxrho if maxrho is not None else float('nan'):.3f} {flag} "
              + " ".join(f"{k}={v:.2f}" for k, v in sorted(per.items()) if v is not None and abs(v) >= 0.3))

    print("\n=== Per-year h10 IC (robustness) ===")
    fwd = closes.shift(-10) / closes - 1.0
    for fid in results:
        fv = cands[fid]
        yrs = {}
        for dt in fv.index:
            if dt > pd.Timestamp(FACTOR_LAST):
                continue
            f = fv.loc[dt]; r = fwd.loc[dt]
            m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
            if m.sum() < MIN_ASSETS:
                continue
            ic, _ = spearmanr(f[m], r[m])
            yrs.setdefault(dt.year, []).append(ic)
        parts = []
        for y in sorted(yrs):
            arr = np.array(yrs[y])
            parts.append(f"{y}: ic={arr.mean():+.4f} icir={arr.mean()/arr.std() if arr.std()>0 else 0:+.3f} n={len(arr)}")
        print(f"{fid:24s} " + " | ".join(parts))
