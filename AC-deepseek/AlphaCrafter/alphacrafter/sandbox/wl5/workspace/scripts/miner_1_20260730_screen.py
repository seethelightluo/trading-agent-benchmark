"""miner_1 factor evaluation framework - batch candidate factor screening.
Uses API data only (respects sim current date). Rank IC / ICIR / turnover / decay on the
15-asset cross-asset universe. Admission gate: |IC|>=0.0070, |ICIR|>=0.0840 (10d horizon)."""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = {"DXY","USDCNY","USDJPY","EURUSD","VIX"}

def load_asset(s, days=1850):
    df = get_stock_daily_data(s, days=days)
    if df is None or len(df) < 120:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["ret"] = df["close"].pct_change()
    return df

def load_macro(s, days=1850):
    df = get_index_daily_data(s, days=days)
    if df is None or len(df) < 120:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["ret"] = df["close"].pct_change()
    return df

# ---------------- candidate factor calculators (return pd.Series indexed by date) ----------------
def rolling_beta(y, x, win):
    """rolling beta of y on x over trailing win (requires aligned series)."""
    cov = y.rolling(win).cov(x)
    var = x.rolling(win).var()
    return cov / var

def factor_high_52w(df, **p):
    return df["close"] / df["close"].rolling(p["win"], min_periods=p["win"]//2).max() - 1.0

def factor_trend_r2(df, **p):
    logp = np.log(df["close"])
    win = p["win"]
    out = pd.Series(np.nan, index=df.index)
    x = np.arange(win)
    xm = x.mean(); xd = ((x - xm) ** 2).sum()
    for i in range(win - 1, len(df)):
        y = logp.iloc[i - win + 1:i + 1].values
        slope = ((x - xm) * (y - y.mean())).sum() / xd
        pred = y.mean() + slope * (x - xm)
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        out.iloc[i] = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else np.nan
    return out

def factor_downside_vol(df, **p):
    neg = df["ret"].clip(upper=0)
    return neg.rolling(p["win"], min_periods=p["win"]//2).std()

def factor_skew(df, **p):
    return df["ret"].rolling(p["win"], min_periods=p["win"]//2).skew()

def factor_max_dd(df, **p):
    return df["close"] / df["close"].rolling(p["win"], min_periods=p["win"]//2).max() - 1.0

def factor_eff_ratio(df, **p):
    num = (df["close"] - df["close"].shift(p["win"])).abs()
    den = df["ret"].abs().rolling(p["win"], min_periods=p["win"]//2).sum()
    return num / den

def factor_autocorr(df, **p):
    return df["ret"].rolling(p["win"], min_periods=p["win"]//2).apply(
        lambda a: pd.Series(a).autocorr(1) if len(a) >= 8 and pd.Series(a).std() > 1e-14 else np.nan, raw=False)

def factor_sharpe(df, **p):
    m = df["ret"].rolling(p["win"], min_periods=p["win"]//2).mean()
    s = df["ret"].rolling(p["win"], min_periods=p["win"]//2).std()
    return m / s

def factor_ts_mom(df, **p):
    return df["close"].shift(p["skip"]) / df["close"].shift(p["lookback"] + p["skip"]) - 1.0

def factor_cond_vix_mom(df, macro, **p):
    """medium momentum only active when VIX below its 60d median, else 0."""
    mom = df["close"].shift(5) / df["close"].shift(25) - 1.0
    vix = macro["VIX"]["close"].reindex(df.index).ffill()
    cond = (vix < vix.rolling(60, min_periods=30).median()).astype(float)
    return mom * cond.fillna(0.0)

def factor_beta_cond(df, macro, mkey, **p):
    """beta(asset, macro_ret, beta_win) * macro momentum (mwin)."""
    m = macro[mkey]
    mret = m["ret"].reindex(df.index).ffill()
    beta = rolling_beta(df["ret"], mret, p["beta_win"])
    mmom = m["close"] / m["close"].shift(p["mwin"]) - 1.0
    mmom = mmom.reindex(df.index).ffill()
    return beta * mmom

def factor_rate_beta(df, macro, **p):
    """beta of asset returns to daily change in US10Y yield."""
    us10y = macro["US10Y"] if "US10Y" in macro else None
    if us10y is None:
        return pd.Series(np.nan, index=df.index)
    dy = us10y["close"].diff().reindex(df.index).ffill()
    return rolling_beta(df["ret"], dy, p["win"])

def factor_range_vol(df, **p):
    rng = (df["high"] - df["low"]) / df["close"]
    return rng.rolling(p["win"], min_periods=p["win"]//2).mean()

def factor_vol_ratio(df, **p):
    """short/long vol ratio (vol compression/expansion)."""
    sv = df["ret"].rolling(p["short"], min_periods=p["short"]//2).std()
    lv = df["ret"].rolling(p["long"], min_periods=p["long"]//2).std()
    return sv / lv

def factor_btc_beta(df, macro, **p):
    ref = macro.get("BTC")
    if ref is None or df.index.name != "date":
        return pd.Series(np.nan, index=df.index)
    r = ref["ret"].reindex(df.index).ffill()
    if df.name == "BTC":
        # use ETH as reference for BTC itself
        eth = macro.get("ETH")
        r = eth["ret"].reindex(df.index).ffill() if eth is not None else r
    return rolling_beta(df["ret"], r, p["win"])

def factor_mom_vix_interaction(df, macro, **p):
    """momentum scaled by VIX z-score (risk-on/off amplification)."""
    mom = df["close"].shift(5) / df["close"].shift(25) - 1.0
    vix = macro["VIX"]["close"].reindex(df.index).ffill()
    vz = (vix - vix.rolling(120, min_periods=60).mean()) / vix.rolling(120, min_periods=60).std()
    return mom * vz.clip(-2, 2)

def factor_xau_ratio(df, macro, **p):
    """asset vs gold relative strength over win days."""
    xau = macro.get("XAU")
    if xau is None:
        return pd.Series(np.nan, index=df.index)
    g = xau["close"].reindex(df.index).ffill()
    return (df["close"] / df["close"].shift(p["win"])) / (g / g.shift(p["win"])) - 1.0

CANDIDATES = {
    "high_52w_dist":      (factor_high_52w,    {"win": 252}, ["close"]),
    "trend_r2_60":        (factor_trend_r2,    {"win": 60},  ["close"]),
    "downside_vol_60":    (factor_downside_vol,{"win": 60},  ["close"]),
    "skew_60":            (factor_skew,        {"win": 60},  ["close"]),
    "max_dd_60":          (factor_max_dd,      {"win": 60},  ["close"]),
    "eff_ratio_20":       (factor_eff_ratio,   {"win": 20},  ["close"]),
    "autocorr_10":        (factor_autocorr,    {"win": 10},  ["close"]),
    "sharpe_60":          (factor_sharpe,      {"win": 60},  ["close"]),
    "ts_mom_60_skip5":    (factor_ts_mom,      {"lookback": 60, "skip": 5}, ["close"]),
    "ts_mom_20_skip5":    (factor_ts_mom,      {"lookback": 20, "skip": 5}, ["close"]),
    "cond_vix_mom_20":    (factor_cond_vix_mom,{}, ["close","VIX"]),
    "dxy_beta_cond_60x20":(factor_beta_cond,   {"beta_win": 60, "mwin": 20}, ["close","DXY"]),
    "usdjpy_beta_cond_60x20":(factor_beta_cond,{"beta_win": 60, "mwin": 20}, ["close","USDJPY"]),
    "eurusd_beta_cond_60x20":(factor_beta_cond,{"beta_win": 60, "mwin": 20}, ["close","EURUSD"]),
    "rate_beta_60":       (factor_rate_beta,   {"win": 60},  ["close","US10Y"]),
    "range_vol_20":       (factor_range_vol,   {"win": 20},  ["close","high","low"]),
    "vol_ratio_10x60":    (factor_vol_ratio,   {"short": 10, "long": 60}, ["close"]),
    "btc_beta_60":        (factor_btc_beta,    {"win": 60},  ["close","BTC","ETH"]),
    "mom_vix_inter_20":   (factor_mom_vix_interaction, {}, ["close","VIX"]),
    "xau_relative_20":    (factor_xau_ratio,   {"win": 20},  ["close","XAU"]),
}

def evaluate_factor(fseries, closes, horizons=(1,2,3,5,10,20), min_assets=8):
    """fseries: dict asset->pd.Series (factor values); closes: dict asset->pd.Series (close)."""
    # align all on union of dates
    panel = pd.DataFrame({a: fseries[a] for a in fseries if fseries[a] is not None})
    cl = pd.DataFrame({a: closes[a] for a in closes if closes[a] is not None})
    results = {}
    ic_by_h = {}
    for H in horizons:
        fwd = cl.shift(-H) / cl - 1.0
        # align dates
        common = panel.index.intersection(fwd.index)
        ics = []
        for dt in common:
            f = panel.loc[dt].dropna()
            r = fwd.loc[dt].reindex(f.index).dropna()
            idx = f.index.intersection(r.index)
            if len(idx) < min_assets:
                continue
            fi, ri = f.loc[idx], r.loc[idx]
            if fi.std() <= 1e-14 or ri.std() <= 1e-14:
                continue
            ic = fi.corr(ri, method="spearman")
            if not math.isnan(ic):
                ics.append(ic)
        ics = np.array(ics)
        ic = float(ics.mean()) if len(ics) else 0.0
        icir = float(ics.mean() / ics.std(ddof=1)) if len(ics) > 2 and ics.std(ddof=1) > 1e-12 else 0.0
        hit = float((ics > 0).mean()) if len(ics) else 0.0
        ic_by_h[int(H)] = {"ic": round(ic, 4), "icir": round(icir, 4), "hit": round(hit, 3), "n": int(len(ics))}
    # coverage
    valid = panel.notna()
    cov_asset_days = float(valid.sum().sum() / max(1, valid.shape[0] * valid.shape[1]))
    cov_dates_ge8 = float((valid.sum(axis=1) >= min_assets).mean())
    # turnover: rank change at 10-day spaced dates
    rpanel = panel.rank(axis=1, pct=True)
    dates = panel.index
    dec_dates = dates[::10]
    turns = []
    for i in range(1, len(dec_dates)):
        a = rpanel.loc[dec_dates[i-1]].dropna()
        b = rpanel.loc[dec_dates[i]].reindex(a.index).dropna()
        idx = a.index.intersection(b.index)
        if len(idx) >= min_assets:
            turns.append(float((a.loc[idx] - b.loc[idx]).abs().mean()))
    turnover = float(np.mean(turns)) if turns else np.nan
    return ic_by_h, cov_asset_days, cov_dates_ge8, turnover

def main():
    assets = {s: load_asset(s) for s in WATCH}
    assets = {s: df for s, df in assets.items() if df is not None and len(df) > 300}
    macro = {s: load_macro(s) for s in MACRO}
    macro = {s: df for s, df in macro.items() if df is not None}
    # also give macro BTC/ETH/XAU/US10Y for reference factors
    macro["BTC"] = assets.get("BTC")
    macro["ETH"] = assets.get("ETH")
    macro["XAU"] = assets.get("XAU")
    macro["US10Y"] = assets.get("US10Y")

    closes = {s: df["close"] for s, df in assets.items()}
    print(f"assets used: {len(assets)}; date range {min(c.index.min() for c in closes.values())}..{max(c.index.max() for c in closes.values())}")

    # library factor values for correlation check
    lib_defs = {
        "mom_10d_skip5":  lambda df: df["close"].shift(5) / df["close"].shift(15) - 1.0,
        "mom_120d_skip5": lambda df: df["close"].shift(5) / df["close"].shift(125) - 1.0,
        "vol_of_vol20x60": lambda df: df["ret"].rolling(20).std().rolling(60).std(),
        "vix_beta_cond_60x20": None,  # needs macro; handle below
    }
    lib_vals = {}
    for name, fn in lib_defs.items():
        if fn is None:
            continue
        lib_vals[name] = {s: fn(df) for s, df in assets.items()}

    # vix_beta_cond: -beta(asset, VIX_ret,60) * vix_mom20
    vix_ret = macro["VIX"]["ret"]
    vix_mom = macro["VIX"]["close"] / macro["VIX"]["close"].shift(20) - 1.0
    lib_vals["vix_beta_cond_60x20"] = {}
    for s, df in assets.items():
        vr = vix_ret.reindex(df.index).ffill()
        beta = rolling_beta(df["ret"], vr, 60)
        mm = vix_mom.reindex(df.index).ffill()
        lib_vals["vix_beta_cond_60x20"][s] = -beta * mm

    out = {}
    for fid, (fn, params, deps) in CANDIDATES.items():
        fseries = {}
        for s, df in assets.items():
            try:
                if "macro" in fn.__code__.co_varnames or any(d in MACRO or d in ("BTC","ETH","XAU","US10Y") for d in deps):
                    fseries[s] = fn(df, macro=macro, **params)
                else:
                    fseries[s] = fn(df, **params)
            except Exception as e:
                fseries[s] = None
        ic_by_h, cov_ad, cov_d8, turn = evaluate_factor(fseries, closes)
        ic10 = ic_by_h[10]["ic"]; icir10 = ic_by_h[10]["icir"]
        # library correlation (flattened standardized factor values)
        max_rho = 0.0
        cand_flat = pd.concat({s: fseries[s] for s in fseries if fseries[s] is not None}).dropna()
        if len(cand_flat) > 100:
            cand_z = (cand_flat - cand_flat.mean()) / cand_flat.std()
            for lname, lvals in lib_vals.items():
                lib_flat = pd.concat({s: lvals[s] for s in lvals if lvals[s] is not None}).dropna()
                if len(lib_flat) < 100:
                    continue
                common = cand_flat.index.intersection(lib_flat.index)
                if len(common) > 100:
                    a = cand_z.loc[common].values
                    b = ((lib_flat.loc[common] - lib_flat.loc[common].mean()) / lib_flat.loc[common].std()).values
                    rho = float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 1e-12 else 0.0
                    max_rho = max(max_rho, abs(rho))
        gate = (abs(ic10) >= 0.0070 and abs(icir10) >= 0.0840)
        out[fid] = {"ic10": round(ic10,4), "icir10": round(icir10,4), "hit10": ic_by_h[10]["hit"],
                    "n_dates": ic_by_h[10]["n"], "cov_ad": round(cov_ad,3), "cov_d8": round(cov_d8,3),
                    "turn": round(turnover,3) if turnover==turnover else None,
                    "max_rho_lib": round(max_rho,3), "decay": {h: v["ic"] for h, v in ic_by_h.items()},
                    "gate_pass": gate}
        flag = "PASS" if gate else "fail"
        print(f"[{flag}] {fid}: IC10={ic10:+.4f} ICIR10={icir10:+.3f} hit={ic_by_h[10]['hit']:.3f} "
              f"n={ic_by_h[10]['n']} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={turnover if turnover==turnover else float('nan'):.2f} rho_lib={max_rho:.3f}")
        print(f"      decay(1,2,3,5,10,20)={[ic_by_h[h]['ic'] for h in (1,2,3,5,10,20)]}")
    with open("scripts/miner_1_20260730_screen_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("saved screen results.")

if __name__ == "__main__":
    main()
