"""
miner_2 cycle 2027-12-21: screen novel factor candidates (data thru 2027-12-20).
Admission window (warm-up): 2020-01-01..2026-07-15. Live drift 2026-07-16..2027-12-20
informational. 15-instrument tradable cross-asset universe (>=8 assets per IC date).

Regime context (12-07 ensemble note): VIX 21.67 re-escalating (28->18 fade REVERSED),
mkt_20 -1.9% risk-off trigger HIT, HIGH dispersion 23.1pp, bonds below MA20 (US10Y -5.4%
last block drag), crypto whipsaw (ETH -18%), N225 bounce. New candidates target:
yield-regime conditioning (bonds below MA20), commodity/safe-haven regime betas,
tail/asymmetry (skew, up/down capture), short-term reversal (momentum whipsaw), and
vol-timing (vol z-score, noise ratio, range position).

Admission gates: |IC_h10| >= 0.007, |ICIR_h10| >= 0.084. Report max_abs_library_correlation
vs the 8 effective library factors (rel_mom_20d_skip5, downside_vol_ratio_20, beta_ew_60d,
corr_ew_60, kurt_20d_skip5, max_ret_20d, dxy_beta_cond_60x20, eurusd_beta_cond_60x20).
"""
from __future__ import annotations
import json, zlib, base64, io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
VISIBLE = "2027-12-20"
WARM_END = "2026-07-15"
LIVE_START = "2026-07-16"
MIN_ASSETS = 8
IC_GATE, ICIR_GATE, CORR_GATE = 0.007, 0.084, 0.5

LIB_IDS = ["rel_mom_20d_skip5", "downside_vol_ratio_20", "beta_ew_60d", "corr_ew_60",
           "kurt_20d_skip5", "max_ret_20d", "dxy_beta_cond_60x20", "eurusd_beta_cond_60x20"]


def load_panel(assets=None):
    assets = assets or WATCH
    closes = {}
    for s in assets:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        closes[s] = df["close"].astype(float)
    panel = pd.concat(closes, axis=1, sort=True)
    return panel[~panel.index.duplicated(keep="last")].sort_index()


def load_macro(name=None):
    out = {}
    for m in (MACRO if name is None else [name]):
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        out[m] = df["close"].astype(float)
    return out


def per_asset(fn):
    def wrapper(panel):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


def fwd_returns(panel, h):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def rank_ic_series(factor, fwd):
    ics = {}
    idx = factor.index.intersection(fwd.index)
    for d in idx:
        f = factor.loc[d].dropna()
        r = fwd.loc[d].reindex(f.index).dropna()
        if len(r) < MIN_ASSETS:
            continue
        ics[d] = spearmanr(f.reindex(r.index), r)[0]
    return pd.Series(ics).sort_index()


def turnover_10d_rank(factor):
    ranks = factor.rank(axis=1)
    out = []
    for i in range(10, len(ranks)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) < MIN_ASSETS:
            continue
        out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


# ---------------- effective library signals (8) ----------------
def library_signals(panel, macro):
    lib = {}
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(panel)
    lib["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    ew = panel.mean(axis=1)
    ew_r = ew.pct_change()

    def ew_beta(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    lib["beta_ew_60d"] = per_asset(ew_beta)(panel)

    def dsvr(s):
        rr = s.pct_change()
        down = rr.where(rr < 0, 0.0)
        ds = np.sqrt((down ** 2).rolling(20).mean())
        tot = rr.rolling(20).std()
        return -(ds / tot)
    lib["downside_vol_ratio_20"] = per_asset(dsvr)(panel)
    lib["max_ret_20d"] = panel.pct_change().rolling(20).max()

    def ew_corr(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).corr(z["m"])
    lib["corr_ew_60"] = per_asset(ew_corr)(panel)

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(panel)

    def fx_cond(ref):
        ref20 = (ref / ref.shift(20) - 1.0)
        def f(s):
            z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
            return beta * ref20.reindex(s.index)
        return per_asset(f)(panel)
    lib["dxy_beta_cond_60x20"] = fx_cond(macro["DXY"].dropna())
    lib["eurusd_beta_cond_60x20"] = fx_cond(macro["EURUSD"].dropna())
    return lib


# ---------------- novel candidates ----------------
def cand_skew_20d_skip5(panel):
    """20d return skewness (skip 5) - tail asymmetry / lottery tilt."""
    def f(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).skew()
    return per_asset(f)(panel)


def cand_reversal_5d_skip1(panel):
    """5d return skip1 (short-term reversal)."""
    return per_asset(lambda s: s.shift(1) / s.shift(6) - 1.0)(panel)


def cand_ref_beta_cond(panel, ref, ref_mom_window=20, beta_window=60):
    """Generic: asset beta to ref returns * ref 20d momentum (regime conditioning)."""
    ref = ref.dropna()
    ref20 = (ref / ref.shift(ref_mom_window) - 1.0)
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for a in panel.columns:
        s = panel[a].dropna()
        z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
        beta = z["r"].rolling(beta_window).cov(z["x"]) / z["x"].rolling(beta_window).var()
        out[a] = (beta * ref20.reindex(s.index)).reindex(panel.index)
    return out


def cand_gk_noise_ratio_20(panel):
    """Garman-Klass vol / close-close vol: noise vs trend efficiency."""
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for a in panel.columns:
        df = panel[a].dropna()
        lo = np.log(df).diff()
        hi = np.log(df).diff()
        # approximate with close-based proxy using high/low if available is not in panel; use returns only
        rr = df.pct_change()
        cc = rr.rolling(20, min_periods=12).std()
        gk = np.sqrt(0.5 * (np.log(df / df.shift(1)) ** 2)).rolling(20, min_periods=12).mean()
        out[a] = (gk / (cc + 1e-12)).reindex(panel.index)
    return out


def cand_updown_ratio_60(panel):
    """60d upside capture / downside capture asymmetry."""
    def f(s):
        rr = s.pct_change()
        up = rr.where(rr > 0, 0.0)
        dn = rr.where(rr < 0, 0.0)
        up_m = up.rolling(60, min_periods=30).mean()
        dn_m = dn.rolling(60, min_periods=30).mean().abs()
        return up_m / (dn_m + 1e-12)
    return per_asset(f)(panel)


def cand_vol_zscore_20x60(panel):
    """(20d vol - rolling mean 60d of 20d vol) / std: vol regime timing."""
    def f(s):
        rr = s.pct_change()
        v20 = rr.rolling(20, min_periods=12).std()
        mu = v20.rolling(60, min_periods=30).mean()
        sd = v20.rolling(60, min_periods=30).std()
        return (v20 - mu) / (sd + 1e-12)
    return per_asset(f)(panel)


def cand_range_pos_20(panel):
    """Normalized position within 20d high-low range."""
    hi20 = panel.rolling(20, min_periods=10).max()
    lo20 = panel.rolling(20, min_periods=10).min()
    return (panel - lo20) / (hi20 - lo20 + 1e-12)


def cand_dd_depth_60(panel):
    """Drawdown depth from 60d high (contrarian oversold signal)."""
    hi60 = panel.rolling(60, min_periods=30).max()
    return panel / hi60 - 1.0


def cand_vol_trend_ratio_10x60(panel):
    """10d vol / 60d vol: short vs long vol regime."""
    def f(s):
        rr = s.pct_change()
        return rr.rolling(10, min_periods=8).std() / (rr.rolling(60, min_periods=30).std() + 1e-12)
    return per_asset(f)(panel)


# ---------------- validation ----------------
def stacked_corr(factor, libsig):
    out = {}
    f = factor.stack().rename("f")
    for fid, sig in libsig.items():
        s = sig.stack().rename("x")
        j = pd.concat([f, s], axis=1).dropna()
        if len(j) > 100:
            out[fid] = float(j["f"].corr(j["x"]))
    return out


def validate(name, factor, panel, libsig, window_end=WARM_END):
    res = {}
    fw = factor.loc[:window_end]
    res["n_dates"] = int(fw.shape[0])
    fwd10 = fwd_returns(panel, 10)
    ic = rank_ic_series(fw, fwd10)
    direction = 1.0 if ic.mean() >= 0 else -1.0
    res["ic_h10"] = float(direction * ic.mean())
    res["icir_h10"] = float(direction * ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
    res["hit_h10"] = float((direction * ic > 0).mean()) if len(ic) else float("nan")
    res["n_h10"] = len(ic)
    res["direction"] = direction
    res["decay"] = {}
    for h in (1, 2, 3, 5, 10, 20):
        ic_h = rank_ic_series(fw, fwd_returns(panel, h))
        res["decay"][str(h)] = float(direction * ic_h.mean()) if len(ic_h) else float("nan")
    valid = fw.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank(fw)
    corrs = stacked_corr(fw, libsig)
    res["max_abs_library_correlation"] = max((abs(v) for v in corrs.values()), default=float("nan"))
    res["library_corrs"] = {k: round(v, 3) for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))}
    gate = abs(res["ic_h10"]) >= IC_GATE and abs(res["icir_h10"]) >= ICIR_GATE
    lowcorr = res["max_abs_library_correlation"] < CORR_GATE
    res["PASS"] = bool(gate and lowcorr)
    res["gate_ic_pass"] = bool(abs(res["ic_h10"]) >= IC_GATE)
    res["gate_icir_pass"] = bool(abs(res["icir_h10"]) >= ICIR_GATE)
    # live drift
    sub = factor.loc[LIVE_START:]
    if sub.notna().sum().sum() >= 100:
        ic_l = rank_ic_series(sub, fwd10.loc[LIVE_START:])
        if len(ic_l) >= 5:
            d = 1.0 if ic_l.mean() >= 0 else -1.0
            res["live_ic_h10"] = float(d * ic_l.mean())
            res["live_icir_h10"] = float(d * ic_l.mean() / ic_l.std()) if ic_l.std() > 0 else float("nan")
            res["live_n"] = len(ic_l)
    print(f"=== {name} === dates={res['n_dates']} direction={direction:+.2f}")
    print(f"  h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} hit={res['hit_h10']:.3f} n={res['n_h10']}")
    print(f"  decay={res['decay']}")
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.3f}")
    print(f"  max_lib_corr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    if "live_ic_h10" in res:
        print(f"  LIVE h10 IC={res['live_ic_h10']:+.4f} ICIR={res['live_icir_h10']:+.4f} n={res['live_n']}")
    print(f"  gate: IC {'OK' if res['gate_ic_pass'] else 'FAIL'} | ICIR {'OK' if res['gate_icir_pass'] else 'FAIL'} | corr<0.5 {'OK' if lowcorr else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}\n")
    return res


def make_artifact(factor):
    """base64:zlib:csv signal panel (warm-up window rows)."""
    csv = factor.round(8).to_csv().encode("utf-8")
    comp = zlib.compress(csv, level=6)
    return base64.b64encode(comp).decode("ascii")


if __name__ == "__main__":
    panel = load_panel()
    macro = load_macro()
    libsig = library_signals(panel, macro)
    print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets; data end {panel.index[-1].date()}")
    print(f"warm-up end {WARM_END}; live {LIVE_START}..{VISIBLE}; library: {list(libsig.keys())}\n")

    r = panel.pct_change()
    print("regime sanity:", end=" ")
    for s in ["VIX", "DXY", "EURUSD", "USDJPY"]:
        if s in macro:
            v = macro[s]
            print(f"{s} last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.1%}", end=" ")
    print()
    for s in WATCH:
        if s in panel.columns and len(r[s].dropna()) >= 60:
            print(f"{s} 20d={r[s].iloc[-20:].add(1).prod()-1:+.1%} 60d={r[s].iloc[-60:].add(1).prod()-1:+.1%}")
    print()

    cands = {
        "skew_20d_skip5": lambda: cand_skew_20d_skip5(panel),
        "reversal_5d_skip1": lambda: cand_reversal_5d_skip1(panel),
        "us10y_beta_cond_60x20": lambda: cand_ref_beta_cond(panel, panel["US10Y"].dropna()),
        "cn10y_beta_cond_60x20": lambda: cand_ref_beta_cond(panel, panel["CN10Y"].dropna()),
        "xau_beta_cond_60x20": lambda: cand_ref_beta_cond(panel, panel["XAU"].dropna()),
        "copper_beta_cond_60x20": lambda: cand_ref_beta_cond(panel, panel["COPPER"].dropna()),
        "gk_noise_ratio_20": lambda: cand_gk_noise_ratio_20(panel),
        "updown_ratio_60": lambda: cand_updown_ratio_60(panel),
        "vol_zscore_20x60": lambda: cand_vol_zscore_20x60(panel),
        "range_pos_20": lambda: cand_range_pos_20(panel),
        "dd_depth_60": lambda: cand_dd_depth_60(panel),
        "vol_trend_ratio_10x60": lambda: cand_vol_trend_ratio_10x60(panel),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, panel, libsig)
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e} ===\n")

    print("##### SUMMARY #####")
    for name, res_ in results.items():
        live = f" live={res_.get('live_ic_h10', float('nan')):+.4f}" if "live_ic_h10" in res_ else ""
        print(f"{name:<26} IC={res_['ic_h10']:+.4f} ICIR={res_['icir_h10']:+.4f} maxcorr={res_['max_abs_library_correlation']:.3f} cov_ge8={res_['coverage_dates_ge8']:.2f} turn={res_['turnover_10d_rank']:.2f}{live} -> {'PASS' if res_['PASS'] else 'FAIL'}")

    Path("scripts/miner_2_20271221_screen_results.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("library_corrs", "decay")}
                    for k, v in results.items()}, indent=2, default=str))
