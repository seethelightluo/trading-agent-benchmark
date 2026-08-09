"""miner_3 factor evaluation harness.

Loads the 15-asset close panel (common dates, through visible cutoff) plus macro
signals, and evaluates candidate factors with cross-sectional IC/ICIR.

IC horizon default h=10 (decision cycle). Spearman IC per common date with >=8
valid assets. ICIR = mean(IC)/std(IC) (daily paper ICIR, non-annualized).
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
CUTOFF = pd.Timestamp("2026-07-15")
MIN_ASSETS = 8

_closes = None
_macro = None

def load_closes():
    global _closes
    if _closes is None:
        _closes = {}
        for a in WATCH:
            df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] <= CUTOFF].set_index("date").sort_index()
            _closes[a] = df
    return _closes

def load_macro():
    global _macro
    if _macro is None:
        _macro = {}
        for a in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
            df = pd.read_csv(f"../persistent/index_data/{a}.csv")
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] <= CUTOFF].set_index("date").sort_index()
            _macro[a] = df
    return _macro

def get_panels():
    """Return close panel, pct-change panel (common dates), macro frames."""
    data = load_closes()
    closes = pd.concat({a: d["close"] for a, d in data.items()}, axis=1, join="inner").dropna()
    rets = closes.pct_change()
    # OHLC panels
    ohlc = {a: d for a, d in data.items()}
    macro = load_macro()
    mclose = {m: d["close"] for m, d in macro.items()}
    macro_panel = pd.concat(mclose, axis=1).reindex(closes.index).ffill()
    return closes, rets, ohlc, macro_panel

def forward_ret(rets, h):
    # cumulative h-day forward return: close[t+h]/close[t] - 1
    closes = (1 + rets).cumprod()
    fwd = closes.shift(-h) / closes - 1.0
    return fwd

def evaluate(factor_df, rets, h=10, name="factor", verbose=True):
    """factor_df: dates x assets. Returns metrics dict."""
    fwd = forward_ret(rets, h)
    common_idx = factor_df.index.intersection(fwd.index)
    factor_df = factor_df.loc[common_idx]
    fwd = fwd.loc[common_idx]
    ics, pears, n_assets, dates = [], [], [], []
    for t in factor_df.index:
        f = factor_df.loc[t]
        r = fwd.loc[t]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() >= MIN_ASSETS:
            ic, _ = spearmanr(f[mask], r[mask])
            if np.isfinite(ic):
                ics.append(ic)
                pc, _ = pearsonr(f[mask], r[mask])
                pears.append(pc if np.isfinite(pc) else np.nan)
                n_assets.append(int(mask.sum()))
                dates.append(t)
    ics = np.array(ics)
    pears = np.array(pears)
    n_assets = np.array(n_assets)
    if len(ics) == 0:
        return {"name": name, "h": h, "n_dates": 0, "mean_ic": np.nan, "std_ic": np.nan,
                "icir": np.nan, "tstat": np.nan, "hit": np.nan, "mean_pearson": np.nan,
                "coverage": np.nan, "turnover": np.nan}
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1)) if len(ics) > 1 else np.nan
    icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan
    tstat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic and std_ic > 0 else np.nan
    hit = float((ics > 0).mean()) if mean_ic >= 0 else float((ics < 0).mean())
    coverage = float((factor_df.notna().sum(axis=1) >= MIN_ASSETS).mean())
    # turnover: mean abs rank change between consecutive 10-day rebalance dates
    tv = np.nan
    f10 = factor_df.iloc[::10]
    if len(f10) > 2:
        ranks = f10.rank(axis=1)
        tv = float(ranks.diff().abs().mean().mean())
    m = {"name": name, "h": h, "n_dates": len(ics), "mean_ic": mean_ic, "std_ic": std_ic,
         "icir": icir, "tstat": tstat, "hit": hit, "mean_pearson": float(np.nanmean(pears)),
         "coverage": coverage, "turnover": tv,
         "n_assets_mean": float(n_assets.mean())}
    if verbose:
        print(f"[{name}] h={h} dates={m['n_dates']} assets={m['n_assets_mean']:.1f} "
              f"IC={mean_ic:+.4f} ICIR={icir:+.3f} t={tstat:+.1f} hit={hit:.2f} "
              f"pearson={m['mean_pearson']:+.4f} cov={coverage:.2f} turn={tv:.3f}")
    return m

def ic_by_regime(factor_df, rets, macro_panel, h=10, name="factor"):
    """IC split by equity up/down and VIX regime."""
    fwd = forward_ret(rets, h)
    idx = factor_df.index.intersection(fwd.index)
    factor_df = factor_df.loc[idx]; fwd = fwd.loc[idx]
    mkt = rets[WATCH].mean(axis=1).loc[idx]
    vix = macro_panel["VIX"].loc[idx]
    ics = {}
    for t in factor_df.index:
        f, r = factor_df.loc[t], fwd.loc[t]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() >= MIN_ASSETS:
            ic, _ = spearmanr(f[mask], r[mask])
            if np.isfinite(ic):
                ics[t] = ic
    if not ics:
        return
    s = pd.Series(ics)
    mkt_ret = mkt.reindex(s.index)
    up = s[mkt_ret > 0]; dn = s[mkt_ret <= 0]
    vix_med = vix.reindex(s.index).median()
    lv = s[vix.reindex(s.index) <= vix_med]; hv = s[vix.reindex(s.index) > vix_med]
    print(f"  regime[{name}]: mkt_up IC={up.mean():+.4f}(n={len(up)}) mkt_dn IC={dn.mean():+.4f}(n={len(dn)}) | "
          f"vix_low IC={lv.mean():+.4f}(n={len(lv)}) vix_high IC={hv.mean():+.4f}(n={len(hv)})")
    # yearly IC
    yr = s.groupby(s.index.year).mean()
    print("  yearly IC:", {int(k): round(float(v), 4) for k, v in yr.items()})
