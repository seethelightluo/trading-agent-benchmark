"""Shared research framework for miner_3 factor validation.
Loads the 15-asset cross-asset universe (close/OHLC) + 5 observation-only macro
series up to the visible-through date, and computes cross-sectional rank IC
metrics used by the benchmark admission gate.

Gate: abs daily paper IC >= 0.0070 AND abs daily paper ICIR >= 0.0840.
ICIR = mean(IC)/std(IC) over daily cross-sectional rank IC observations.
A date needs >= min_valid (default 8) valid factor values to count.
"""
import json
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
DATA_ROOT = "../persistent/stock_data"
MACRO_ROOT = "../persistent/index_data"

IC_GATE = 0.0070
ICIR_GATE = 0.0840


def load_visible_through():
    d = json.load(open("../persistent/date.json"))
    return d.get("visible_through", d["current_date"])


def load_closes(end_date=None):
    """Return DataFrame (dates x assets) of close prices, sorted, through end_date."""
    if end_date is None:
        end_date = load_visible_through()
    closes = {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_ROOT}/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= end_date].set_index("date")["close"]
        closes[a] = df
    panel = pd.DataFrame(closes).sort_index()
    return panel


def load_ohlc(end_date=None):
    """Return dict asset -> DataFrame(date, open, high, low, close) through end_date."""
    if end_date is None:
        end_date = load_visible_through()
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_ROOT}/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= end_date].set_index("date")
        out[a] = df[["open", "high", "low", "close"]]
    return out


def load_macro(end_date=None):
    if end_date is None:
        end_date = load_visible_through()
    out = {}
    for m in MACRO:
        df = pd.read_csv(f"{MACRO_ROOT}/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= end_date].set_index("date")
        out[m] = df["close"]
    return pd.DataFrame(out).sort_index()


def forward_returns(closes, horizon=10):
    """h-trading-day forward simple return on the union calendar."""
    return closes.shift(-horizon) / closes - 1.0


def rank_ic_series(factor_panel, fwd, min_valid=8):
    """Daily cross-sectional Spearman rank IC between factor and forward return."""
    dates, ics = [], []
    for dt in factor_panel.index:
        f = factor_panel.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < min_valid:
            continue
        ic = f[mask].rank().corr(r[mask].rank())
        if np.isfinite(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")


def summarize_ic(ics, turnover=None, coverage=None, label=""):
    n = len(ics)
    ic_mean = ics.mean() if n else np.nan
    ic_std = ics.std() if n else np.nan
    icir = ic_mean / ic_std if (n and ic_std and ic_std > 0) else np.nan
    hit = (np.sign(ics) == np.sign(ic_mean)).mean() if n else np.nan
    print(f"--- {label} ---")
    print(f"  n_ic_dates={n}  IC={ic_mean:.4f}  ICIR={icir:.4f}  IC_std={ic_std:.4f}  hit={hit:.3f}")
    print(f"  gate: |IC|>={IC_GATE} {'PASS' if abs(ic_mean)>=IC_GATE else 'FAIL'}  "
          f"|ICIR|>={ICIR_GATE} {'PASS' if abs(icir)>=ICIR_GATE else 'FAIL'}")
    if turnover is not None:
        print(f"  turnover_10d_rank={turnover:.3f}")
    if coverage is not None:
        print(f"  coverage_asset_days={coverage:.3f}")
    return {"ic": float(ic_mean), "icir": float(icir), "ic_std": float(ic_std),
            "ic_hit_ratio": float(hit), "n_ic_dates": int(n)}


def turnover_10d_rank(factor_panel):
    """Mean absolute change in cross-sectional rank over 10 trading days."""
    ranks = factor_panel.rank(axis=1)
    d = ranks.diff(10).abs().mean(skipna=True)
    return float(d)


def coverage_asset_days(factor_panel):
    tot = factor_panel.shape[0] * factor_panel.shape[1]
    return float(factor_panel.notna().sum().sum() / tot)


def compute_library_corr(factor_panel):
    """Max abs correlation of factor signal with the 3 effective library factor signals
    (recomputed from their persisted definitions on the same panel)."""
    closes = load_closes()
    rets = closes.pct_change()
    lib = {}
    # vol_adj_mom_accel_20x60
    f1 = (closes / closes.shift(20) - 1 - (closes / closes.shift(60) - 1)) / rets.rolling(20).std()
    lib["vol_adj_mom_accel_20x60"] = f1
    # dn_mkt_beta_60d : beta of asset ret on min(mkt_ret,0), 60d
    mkt = rets.mean(axis=1)
    dm = mkt.where(mkt < 0)
    beta = rets.rolling(60).cov(dm) / dm.rolling(60).var()
    lib["dn_mkt_beta_60d"] = beta
    # rate_beta_cn10y_60d : beta of asset ret on pct_change(CN10Y), 60d
    cn = closes["CN10Y"].pct_change()
    beta2 = rets.rolling(60).cov(cn) / cn.rolling(60).var()
    lib["rate_beta_cn10y_60d"] = beta2

    best = 0.0
    best_name = None
    for name, s in lib.items():
        common = factor_panel.notna() & s.notna()
        if common.sum().sum() < 500:
            continue
        # flatten signals pairwise over common entries
        a = factor_panel[common].to_numpy().ravel()
        b = s[common].to_numpy().ravel()
        rho = np.corrcoef(a, b)[0, 1]
        if np.isfinite(rho) and abs(rho) > best:
            best = abs(rho)
            best_name = name
    return best, best_name


def decay_analysis(factor_panel, closes, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = forward_returns(closes, horizon=h)
        ics = rank_ic_series(factor_panel, fwd)
        out[str(h)] = round(float(ics.mean()), 4) if len(ics) else None
    return out


def run_full_validation(factor_panel, label, horizon=10, verbose=True):
    closes = load_closes()
    fwd = forward_returns(closes, horizon=horizon)
    ics = rank_ic_series(factor_panel, fwd)
    if verbose:
        print(f"factor panel shape={factor_panel.shape}")
        print(f"dates with >=8 valid: {ics.shape[0]} / {factor_panel.shape[0]}")
    res = summarize_ic(ics, label=label)
    res["turnover_10d_rank"] = turnover_10d_rank(factor_panel)
    res["coverage_asset_days"] = coverage_asset_days(factor_panel)
    res["decay_ic_by_horizon"] = decay_analysis(factor_panel, closes)
    best, best_name = compute_library_corr(factor_panel)
    res["max_abs_library_correlation"] = round(best, 4)
    res["max_corr_factor"] = best_name
    if verbose:
        print(f"  max_abs_library_correlation={best:.4f} ({best_name})")
        print(f"  decay_ic: {res['decay_ic_by_horizon']}")
    return res
