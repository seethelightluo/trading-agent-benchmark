"""
miner_3 cycle 2027-01-19: screen round 18.
Re-validates the 8-factor library (warm-up gate + live drift through 2027-01-18)
and screens fresh round-18 candidates. Admission gates (same as prior cycles):
|IC_h10| >= 0.007, |ICIR_h10| >= 0.084, max_abs_library_correlation < 0.5.

Round-18 candidates (fresh, cross-checked against rounds 1-17 / miner_1/2 screens):
  above_ma_ratio_60    - share of last 60 days with close > 20d MA (trend persistence;
                         addresses rel_mom whipsaw flag: momentum in consistent-trend names)
  mom_ratio_20x60      - mom20_skip5 / (|mom60_skip5|+eps): trend stage (early vs mature)
  crypto_link_60       - mean 60d rolling corr of asset returns with BTC & ETH
                         (digital-asset comovement centrality)
  lowvol_mom_20x60     - mom20_skip5 gated by vol20 < vol60 (momentum in calm regimes;
                         addresses high-vol whipsaw flag)
  down_up_beta_ratio_60- downside EW beta / upside EW beta (asymmetric mkt sensitivity)
  corr_dispersion_60   - std of 60d pairwise correlations per asset (comovement heterogeneity)
  gain_concentration_60- share of 60d cumulative gain from top-3 up days (lottery concentration)
  underwater_time_60   - fraction of last 60 days spent below rolling 60d max (time in drawdown)
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MIN_ASSETS = 8
WARM_END = "2026-07-15"
LIVE_START = "2026-07-16"
DAYS = 4000
IC_GATE, ICIR_GATE, CORR_GATE = 0.007, 0.084, 0.5


def load_ohlcv(days=DAYS):
    closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
    for s in WATCH:
        df = get_stock_daily_data(s, days=days)
        if df is None or not len(df):
            continue
        df = df.set_index("date")
        closes[s] = df["close"].astype(float)
        opens[s] = df["open"].astype(float)
        highs[s] = df["high"].astype(float)
        lows[s] = df["low"].astype(float)
        vols[s] = df["volume"].astype(float)

    def _p(d):
        p = pd.concat(d, axis=1, sort=True)
        return p[~p.index.duplicated(keep="last")].sort_index()
    return _p(closes), _p(opens), _p(highs), _p(lows), _p(vols)


def load_macro():
    out = {}
    for s in MACRO:
        df = get_index_daily_data(s, days=DAYS)
        if df is not None and len(df):
            out[s] = df.set_index("date")["close"].astype(float)
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
    f = factor.stack().rename("f")
    r = fwd.stack().rename("r")
    j = pd.concat([f, r], axis=1).dropna()
    if len(j) == 0:
        return pd.Series(dtype=float)
    j["fr"] = j.groupby(level=0)["f"].rank()
    j["rr"] = j.groupby(level=0)["r"].rank()
    cnt = j.groupby(level=0).size()
    keep = cnt[cnt >= MIN_ASSETS].index
    j = j[j.index.get_level_values(0).isin(keep)]
    g = j.groupby(level=0)
    n = g.size()
    sx, sy = g["fr"].sum(), g["rr"].sum()
    sxx, syy = (g["fr"] ** 2).sum(), (g["rr"] ** 2).sum()
    sxy = (g["fr"] * g["rr"]).sum()
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    ic = num / den
    return ic.sort_index()


def turnover_10d_rank(factor):
    ranks = factor.rank(axis=1)
    out, dates = [], ranks.index
    for i in range(10, len(dates)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) >= MIN_ASSETS:
            out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


# ---------------- current library signals (8 kept) ----------------
def library_signals(close, high, low, vol, macro):
    lib = {}
    r = close.pct_change()
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    lib["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()

    def ew_beta(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    lib["beta_ew_60d"] = per_asset(ew_beta)(close)

    def dsvr(s):
        rr = s.pct_change()
        down = rr.where(rr < 0, 0.0)
        ds = np.sqrt((down ** 2).rolling(20).mean())
        tot = rr.rolling(20).std()
        return -(ds / tot)
    lib["downside_vol_ratio_20"] = per_asset(dsvr)(close)
    lib["max_ret_20d"] = r.rolling(20).max()

    def ew_corr(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).corr(z["m"])
    lib["corr_ew_60"] = per_asset(ew_corr)(close)

    def fx_cond(ref):
        ref20 = (ref / ref.shift(20) - 1.0)
        def f(s):
            z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
            return beta * ref20.reindex(s.index)
        return per_asset(f)(close)
    lib["eurusd_beta_cond_60x20"] = fx_cond(macro["EURUSD"].dropna())
    lib["dxy_beta_cond_60x20"] = -fx_cond(macro["DXY"].dropna())

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close)
    return lib


# ---------------- round-18 candidates ----------------
def cand_above_ma_ratio_60(close):
    ma20 = close.rolling(20, min_periods=10).mean()
    above = close > ma20
    return above.rolling(60, min_periods=30).mean()


def cand_mom_ratio_20x60(close):
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    m60 = per_asset(lambda s: s.shift(5) / s.shift(65) - 1.0)(close)
    return m20 / (m60.abs() + 1e-6)


def cand_crypto_link_60(close):
    r = close.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    crypto = [a for a in close.columns if a in ("BTC", "ETH")]
    for a in close.columns:
        parts = []
        for c in crypto:
            if c == a:
                continue
            parts.append(r[a].rolling(60, min_periods=30).corr(r[c]))
        if parts:
            out[a] = pd.concat(parts, axis=1).mean(axis=1)
        else:
            out[a] = np.nan
    return out


def cand_lowvol_mom_20x60(close):
    r = close.pct_change()
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    v20 = r.rolling(20, min_periods=10).std()
    v60 = r.rolling(60, min_periods=30).std()
    gate = (v20 < v60).astype(float)
    return m20 * gate


def cand_down_up_beta_ratio_60(close):
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = close[a].dropna()
        rr = s.pct_change()
        z = pd.concat([rr.rename("r"), ew_r.rename("m")], axis=1).dropna()
        down = z[z["m"] < 0]
        up = z[z["m"] > 0]
        dbeta = down["r"].rolling(60, min_periods=20).cov(down["m"]) / down["m"].rolling(60, min_periods=20).var()
        ubeta = up["r"].rolling(60, min_periods=20).cov(up["m"]) / up["m"].rolling(60, min_periods=20).var()
        ratio = dbeta / (ubeta.abs() + 1e-6)
        out[a] = ratio.reindex(close.index)
    return out


def cand_corr_dispersion_60(close):
    r = close.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        peers = [b for b in close.columns if b != a]
        corrs = pd.concat([r[a].rolling(60, min_periods=30).corr(r[b]).rename(b) for b in peers], axis=1)
        out[a] = corrs.std(axis=1, ddof=0)
    return out


def cand_gain_concentration_60(close):
    r = close.pct_change()
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = r[a].dropna()
        total = s.rolling(60, min_periods=30).sum().abs().replace(0, np.nan)
        top3 = s.rolling(60, min_periods=30).apply(lambda w: w.clip(lower=0).nlargest(3).sum(), raw=True)
        out[a] = (top3 / total).reindex(close.index)
    return out


def cand_underwater_time_60(close):
    max60 = close.rolling(60, min_periods=30).max()
    below = close < max60
    return below.rolling(60, min_periods=30).mean()


# ---------------- validation ----------------
def stacked_corr(cand, libsig):
    out = {}
    f = cand.stack().rename("f")
    for fid, ls in libsig.items():
        g = ls.stack().rename("g")
        j = pd.concat([f, g], axis=1).dropna()
        if len(j) < 100:
            out[fid] = float("nan")
            continue
        out[fid] = float(j["f"].corr(j["g"], method="spearman")) if np.isfinite(j["f"].corr(j["g"], method="spearman")) else float("nan")
    return out


def validate(name, factor, close, libsig, window_end=WARM_END):
    res = {"n_dates": int(factor.loc[:window_end].shape[0])}
    fwd10 = fwd_returns(close, 10)
    ic = rank_ic_series(factor.loc[:window_end], fwd10)
    direction = 1.0 if ic.mean() >= 0 else -1.0
    res["ic_h10"] = float(direction * ic.mean())
    res["icir_h10"] = float(direction * ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
    res["hit_h10"] = float((direction * ic > 0).mean()) if len(ic) else float("nan")
    res["n_h10"] = len(ic)
    res["direction"] = direction
    res["decay"] = {}
    for h in (1, 2, 3, 5, 10, 20):
        ic_h = rank_ic_series(factor.loc[:window_end], fwd_returns(close, h))
        res["decay"][str(h)] = float(direction * ic_h.mean()) if len(ic_h) else float("nan")
    valid = factor.loc[:window_end].notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    res["turnover_10d_rank"] = turnover_10d_rank(factor.loc[:window_end])
    corrs = stacked_corr(factor.loc[:window_end], libsig)
    res["max_abs_library_correlation"] = max((abs(v) for v in corrs.values()), default=float("nan"))
    res["library_corrs"] = {k: round(v, 3) for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))}
    gate = abs(res["ic_h10"]) >= IC_GATE and abs(res["icir_h10"]) >= ICIR_GATE
    lowcorr = res["max_abs_library_correlation"] < CORR_GATE
    res["PASS"] = bool(gate and lowcorr)
    print(f"=== {name} === dates={res['n_dates']} direction={direction:+.2f}")
    print(f"  h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} hit={res['hit_h10']:.3f} n={res['n_h10']}")
    print(f"  decay={res['decay']}")
    print(f"  cov_asset={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.3f}")
    print(f"  max_lib_corr={res['max_abs_library_correlation']:.3f} corrs={res['library_corrs']}")
    print(f"  gate: IC>={IC_GATE} {'OK' if abs(res['ic_h10'])>=IC_GATE else 'FAIL'} | ICIR>={ICIR_GATE} {'OK' if abs(res['icir_h10'])>=ICIR_GATE else 'FAIL'} | corr<{CORR_GATE} {'OK' if lowcorr else 'FAIL'} -> {'PASS' if res['PASS'] else 'FAIL'}\n")
    return res


if __name__ == "__main__":
    close, open_, high, low, vol = load_ohlcv()
    macro = load_macro()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; data end {close.index[-1].date()}; visible_through expected 2027-01-18")
    print(f"library factors: {list(libsig.keys())}\n")

    print("##### A) LIBRARY RE-VALIDATION (warm-up) #####")
    lib_results = {}
    for fid, sig in libsig.items():
        lib_results[fid] = validate(f"[LIB] {fid}", sig, close, libsig)

    print("##### A2) LIVE-WINDOW DRIFT (informational: 2026-07-16..end) #####")
    for fid, sig in libsig.items():
        sub = sig.loc[LIVE_START:]
        if sub.notna().sum().sum() < 200:
            print(f"  {fid}: too few live obs, skip")
            continue
        ic = rank_ic_series(sub, fwd_returns(close, 10).loc[LIVE_START:])
        if len(ic) < 5:
            print(f"  {fid}: live IC n={len(ic)} too small")
            continue
        d = 1.0 if ic.mean() >= 0 else -1.0
        icir = d * ic.mean() / ic.std() if ic.std() > 0 else float("nan")
        print(f"  {fid}: live h10 IC={d*ic.mean():+.4f} ICIR={icir:+.4f} hit={(d*ic>0).mean():.3f} n={len(ic)}")
    print()

    print("##### B) CANDIDATE SCREENS (round 18) #####")
    cands = {
        "above_ma_ratio_60": lambda: cand_above_ma_ratio_60(close),
        "mom_ratio_20x60": lambda: cand_mom_ratio_20x60(close),
        "crypto_link_60": lambda: cand_crypto_link_60(close),
        "lowvol_mom_20x60": lambda: cand_lowvol_mom_20x60(close),
        "down_up_beta_ratio_60": lambda: cand_down_up_beta_ratio_60(close),
        "corr_dispersion_60": lambda: cand_corr_dispersion_60(close),
        "gain_concentration_60": lambda: cand_gain_concentration_60(close),
        "underwater_time_60": lambda: cand_underwater_time_60(close),
    }
    results = {}
    for name, fn in cands.items():
        try:
            factor = fn()
            results[name] = validate(name, factor, close, libsig)
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e} ===\n")

    print("##### SUMMARY #####")
    for name, r in results.items():
        print(f"{name:<24} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} maxcorr={r['max_abs_library_correlation']:.3f} cov_ge8={r['coverage_dates_ge8']:.2f} turn={r['turnover_10d_rank']:.2f} -> {'PASS' if r['PASS'] else 'FAIL'}")
