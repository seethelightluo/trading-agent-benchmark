"""miner_1 cycle: screen novel factor families vs FULL 9-factor library.
Universe: 15 tradable cross-asset instruments. Validation window 2020-01-01..2026-07-15.
Admission gates (benchmark contract): |IC|>=0.007, |ICIR|>=0.084 @ h=10.
Full library correlation audit: recompute all 9 effective factors, mean per-date
cross-sectional Spearman rho over last 700 dates.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import miner_2_lib as lib

panel = lib.load_panel()
macro = lib.load_macro()
WATCH = lib.WATCH
FACTOR_LAST = lib.FACTOR_LAST
MAX_VISIBLE = lib.MAX_VISIBLE
MIN_ASSETS = lib.MIN_ASSETS
ADMISSION = lib.ADMISSION
rets = panel.pct_change()


# ---------------- full 9-factor library recomputation ----------------
def build_full_library():
    libs = {}
    mkt = rets.mean(axis=1)
    # 1 rel_mom_20d_skip5
    raw = panel.shift(5) / panel.shift(25) - 1.0
    libs["rel_mom_20d_skip5"] = raw.sub(raw.median(axis=1), axis=0)
    # 2 beta_ew_60d
    cols = {}
    for a in panel.columns:
        s = rets[a].dropna()
        m = mkt.reindex(s.index)
        z = pd.concat([s.rename("r"), m.rename("m")], axis=1).dropna()
        cols[a] = z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    libs["beta_ew_60d"] = pd.DataFrame(cols, index=panel.index)
    # 3 mom_120d_skip5
    libs["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
    # 4 vol_of_vol20x60
    libs["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    # 5 max_ret_20d
    libs["max_ret_20d"] = rets.rolling(20).max()
    # 6 downside_vol_ratio_20 = -(dn/tot)
    def dsvr(s):
        r = s.pct_change()
        tot = r.rolling(20).std()
        dn = r.where(r < 0, 0.0).rolling(20).std()
        return -(dn / tot)
    cols = {}
    for a in panel.columns:
        cols[a] = dsvr(panel[a].dropna())
    libs["downside_vol_ratio_20"] = pd.DataFrame(cols, index=panel.index)
    # 7 mom_10d_skip5
    libs["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
    # 8 amihud_20
    am = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        r = df["close"].pct_change()
        am[s] = (r.abs() / (df["volume"] + 1e-12)).rolling(20, min_periods=10).mean()
    libs["amihud_20"] = pd.DataFrame(am, index=panel.index)
    # 9 vix_beta_cond_60x20
    vix = macro["VIX"].dropna()
    vixr = vix.pct_change()
    beta = rets.rolling(60).cov(vixr) / vixr.rolling(60).var()
    libs["vix_beta_cond_60x20"] = -beta * (vix / vix.shift(20) - 1.0)
    return libs


def full_library_corr(factor):
    libs = build_full_library()
    per = {}
    common = factor.index.intersection(panel.index)
    for fid, lf in libs.items():
        cs = []
        for dt in common[-700:]:
            if dt not in lf.index:
                continue
            f = factor.loc[dt]
            g = lf.loc[dt]
            if isinstance(f, pd.Series) and isinstance(g, pd.Series):
                m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
                m = m.reindex(f.index).fillna(False)
                if int(m.sum()) >= MIN_ASSETS:
                    cs.append(spearmanr(f[m], g[m])[0])
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in per.values() if v is not None]
    return round(max(valid), 4) if valid else float("nan"), per


# ---------------- candidate factors ----------------
def per_asset_fn(fn):
    def inner(pnl, mcr):
        cols = {}
        for a in pnl.columns:
            s = pnl[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=pnl.index)
    return inner


# 1) Kaufman efficiency ratio 20d: |net move| / gross path
def cand_kaufman_eff(n=20):
    def f(s):
        r = s.pct_change()
        gross = r.abs().rolling(n).sum()
        net = (s / s.shift(n) - 1.0).abs()
        return net / gross
    return per_asset_fn(f)


# 2) AR(1) autocorrelation of daily returns over 60d
def cand_autocorr(win=60):
    def f(s):
        r = s.pct_change()
        def ac(x):
            x = x.dropna()
            if len(x) < 10:
                return np.nan
            a, b = x.iloc[:-1], x.iloc[1:]
            if a.std() == 0 or b.std() == 0:
                return np.nan
            return np.corrcoef(a, b)[0, 1]
        return r.rolling(win).apply(ac, raw=False)
    return per_asset_fn(f)


# 3) Elder force index 20d: rolling mean of volume * pct_change
def cand_force_index(n=20):
    vol = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        vol[s] = df["volume"].astype(float)
    vol = pd.DataFrame(vol, index=panel.index)
    raw = vol * rets
    cols = {}
    for a in panel.columns:
        cols[a] = raw[a].rolling(n, min_periods=10).mean()
    return pd.DataFrame(cols, index=panel.index)


# 4) Money Flow Index 20d (close-based typical price)
def cand_mfi(n=20):
    vol = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        vol[s] = df["volume"].astype(float)
    vol = pd.DataFrame(vol, index=panel.index)
    cols = {}
    for a in panel.columns:
        c = panel[a]
        chg = c.diff()
        pos = (chg.clip(lower=0) * vol[a]).rolling(n, min_periods=10).sum()
        neg = (-chg.clip(upper=0) * vol[a]).rolling(n, min_periods=10).sum()
        mfi = 100.0 - 100.0 / (1.0 + pos / neg.replace(0, np.nan))
        cols[a] = mfi
    return pd.DataFrame(cols, index=panel.index)


# 5) RSI-14
def cand_rsi(n=14):
    def f(s):
        r = s.pct_change()
        up = r.clip(lower=0).rolling(n).mean()
        dn = (-r.clip(upper=0)).rolling(n).mean()
        rs = up / dn.replace(0, np.nan)
        return 100.0 - 100.0 / (1.0 + rs)
    return per_asset_fn(f)


# 6-9) macro beta family
def macro_beta(name, win=60, flip=False):
    mv = macro[name].dropna()
    def f(s):
        r = s.pct_change()
        v = mv.pct_change().reindex(s.index)
        z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
        b = z["r"].rolling(win).cov(z["v"]) / z["v"].rolling(win).var().replace(0, np.nan)
        return -b if flip else b
    return per_asset_fn(f)


# 10) trend acceleration: mom60(skip5) - mom20(skip5)
def cand_accel(n_long=60, n_short=20, skip=5):
    def f(s):
        return (s.shift(skip) / s.shift(n_long + skip) - 1.0) - (s.shift(skip) / s.shift(n_short + skip) - 1.0)
    return per_asset_fn(f)


# 11) 60d Sharpe (mean/std of returns)
def cand_sharpe(n=60):
    def f(s):
        r = s.pct_change()
        return r.rolling(n).mean() / r.rolling(n).std().replace(0, np.nan)
    return per_asset_fn(f)


# 12) up/down vol ratio 20d: mean(pos ret)/mean(|neg ret|)
def cand_up_down_ratio(n=20):
    def f(s):
        r = s.pct_change()
        up = r.clip(lower=0).rolling(n).mean()
        dn = (-r.clip(upper=0)).rolling(n).mean()
        return up / dn.replace(0, np.nan)
    return per_asset_fn(f)


if __name__ == "__main__":
    cands = [
        ("kaufman_eff_20", cand_kaufman_eff(20), ["close"], {"window": 20}),
        ("autocorr_60", cand_autocorr(60), ["close"], {"window": 60}),
        ("force_index_20", cand_force_index(20), ["close", "volume"], {"window": 20}),
        ("mfi_20", cand_mfi(20), ["close", "volume"], {"window": 20}),
        ("rsi_14", cand_rsi(14), ["close"], {"window": 14}),
        ("eurusd_beta_60", macro_beta("EURUSD", 60), ["close", "EURUSD"], {"window": 60}),
        ("usdcny_beta_60", macro_beta("USDCNY", 60), ["close", "USDCNY"], {"window": 60}),
        ("dxy_beta_60", macro_beta("DXY", 60), ["close", "DXY"], {"window": 60}),
        ("vix_beta_raw_60", macro_beta("VIX", 60, flip=True), ["close", "VIX"], {"window": 60}),
        ("accel_60x20", cand_accel(60, 20, 5), ["close"], {"n_long": 60, "n_short": 20, "skip": 5}),
        ("sharpe_60", cand_sharpe(60), ["close"], {"window": 60}),
        ("up_down_ratio_20", cand_up_down_ratio(20), ["close"], {"window": 20}),
    ]

    print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets | "
          f"warmup through {FACTOR_LAST}")
    print("building full 9-factor library...")
    libs = build_full_library()
    print("library:", list(libs.keys()))
    print()

    summary = {}
    for name, fn, deps, params in cands:
        try:
            res = lib.validate_factor(name, fn, horizons=(1, 2, 3, 5, 10, 20))
        except Exception as e:
            print(f"=== {name} === ERROR: {e}\n")
            summary[name] = {"error": str(e)}
            continue
        # full-library correlation
        factor_w = None
        fac = fn(panel, macro)
        maxc, per = full_library_corr(fac.loc[:FACTOR_LAST])
        res["max_abs_library_correlation"] = maxc
        res["library_corrs_full"] = per
        print(f"  FULL-LIB max_abs_corr={maxc:.3f}")
        print()
        summary[name] = {
            "ic10": res["ic_h10"], "icir10": res["icir_h10"],
            "hit10": res["hit_h10"], "n": res["n_dates_h10"],
            "cov_ad": res["coverage_asset_days"], "cov_d8": res["coverage_dates_ge8"],
            "turnover": res["turnover_10d_rank"],
            "max_corr": maxc, "pass": res["admission_gate"]["pass"],
            "decay": res["decay_ic_by_horizon"],
        }

    print("\n================ SUMMARY ================")
    for name, s in summary.items():
        if "error" in s:
            print(f"{name:24s} ERROR {s['error']}")
            continue
        print(f"{name:24s} IC10={s['ic10']:+.4f} ICIR10={s['icir10']:+.4f} "
              f"hit={s['hit10']:.3f} n={s['n']} cov_ad={s['cov_ad']:.3f} "
              f"max_corr={s['max_corr']:.3f} PASS={s['pass']}")
