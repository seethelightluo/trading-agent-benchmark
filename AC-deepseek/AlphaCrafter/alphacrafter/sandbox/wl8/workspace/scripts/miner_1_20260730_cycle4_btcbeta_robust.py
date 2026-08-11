"""miner_1 cycle 4: robustness checks for btc_beta_60 + crypto-beta variants."""
import json
import base64
import zlib
import io
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
SIGNAL_END = pd.Timestamp("2026-07-29")
DATA_END = pd.Timestamp("2026-07-30")
MIN_ASSETS_PER_DATE = 8

closes = {}
for a in ASSETS:
    df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
    df = df[df["date"] <= DATA_END].set_index("date").sort_index()
    closes[a] = df["close"].astype(float)
close = pd.DataFrame(closes)

def f_beta_asset(asset_ret, macro_ret, win=60, minp=None):
    common = asset_ret.index.intersection(macro_ret.index)
    r_a = asset_ret.reindex(common).dropna()
    r_m = macro_ret.reindex(common).dropna()
    idx = r_a.index.intersection(r_m.index)
    r_a, r_m = r_a.reindex(idx), r_m.reindex(idx)
    minp = minp or max(30, win // 2)
    cov = r_a.rolling(win, min_periods=minp).cov(r_m)
    var = r_m.rolling(win, min_periods=minp).var()
    return cov / var

def make_beta_factor(macro_close, win=60, minp=None):
    out = {}
    for a in ASSETS:
        idx = closes[a].dropna().index
        c = closes[a].reindex(idx)
        asset_ret = c.pct_change()
        m_aligned = macro_close.reindex(idx).ffill()
        macro_ret = m_aligned.pct_change()
        out[a] = f_beta_asset(asset_ret, macro_ret, win, minp)
    return pd.DataFrame({a: s.reindex(close.index) for a, s in out.items()})

def fwd_returns(horizon):
    out = {}
    for a in ASSETS:
        c = closes[a].dropna()
        fr = (c.shift(-horizon) / c - 1.0).reindex(close.index)
        out[a] = fr
    return pd.DataFrame(out)

def ic_series(factor, fwd_ret, min_assets=MIN_ASSETS_PER_DATE):
    dates, ics = [], []
    for dt in factor.index:
        x, y = factor.loc[dt], fwd_ret.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= min_assets:
            ics.append(x[m].rank().corr(y[m].rank()))
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def turnover_rank(factor, lag=10):
    ranks = factor.rank(axis=1)
    return float(ranks.diff(lag).abs().mean(axis=1).dropna().mean())

def load_active_library():
    lib = {}
    for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        d = json.load(open(f"factors/{fid}.json"))
        raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
        p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
        lib[fid] = p
    return lib

LIB = load_active_library()

def lib_corr_breakdown(panel):
    out = {}
    for fid, lp in LIB.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        out[fid] = round(float(np.corrcoef(a[m], b[m])[0, 1]), 4) if m.sum() > 200 else None
    return out

def full_validate(panel):
    panel = panel[panel.index <= SIGNAL_END]
    fr = fwd_returns(10)
    ic = ic_series(panel, fr)
    ic_mean = float(ic.mean())
    icir = float(ic.mean() / ic.std()) if len(ic) > 2 else np.nan
    # sub-period stability
    half = len(ic) // 2
    ic1 = float(ic.iloc[:half].mean())
    ic2 = float(ic.iloc[half:].mean())
    recent2y = float(ic[ic.index >= "2024-07-01"].mean()) if (ic.index >= "2024-07-01").any() else np.nan
    return {
        "ic_10d": round(ic_mean, 4), "icir": round(icir, 4),
        "n": int(len(ic)), "ic_first_half": round(ic1, 4),
        "ic_second_half": round(ic2, 4), "ic_recent2y": round(recent2y, 4),
        "turnover_10d": turnover_rank(panel),
        "libcorr": lib_corr_breakdown(panel),
    }

BTC = closes["BTC"]
print("=== robustness: btc_beta_60 (original) ===")
p60 = make_beta_factor(BTC, 60)
print(full_validate(p60))

print("\n=== variants ===")
for name, win, minp in [("btc_beta_90", 90, None), ("btc_beta_120", 120, None),
                        ("btc_beta_60_60min", 60, 60), ("btc_beta_180", 180, None)]:
    p = make_beta_factor(BTC, win, minp)
    r = full_validate(p)
    print(f"{name}: ic={r['ic_10d']} icir={r['icir']} n={r['n']} "
          f"half1={r['ic_first_half']} half2={r['ic_second_half']} recent2y={r['ic_recent2y']} "
          f"turnover={round(r['turnover_10d'],2)} libcorr={r['libcorr']}")

print("\n=== eth_beta_60 (beta to ETH) ===")
ETH = closes["ETH"]
p = make_beta_factor(ETH, 60)
print(full_validate(p))

print("\n=== crypto_avg_beta_60 (beta to BTC+ETH avg) ===")
cavg = pd.concat([BTC, ETH]).groupby(level=0).mean().sort_index()
p = make_beta_factor(cavg, 60)
print(full_validate(p))

# per-factor libcorr for range_pos_60 candidates re-check
print("\n=== correlation of btc_beta_60 with yield_beta_cond over time ===")
