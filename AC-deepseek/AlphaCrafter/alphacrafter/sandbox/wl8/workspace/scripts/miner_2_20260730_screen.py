"""miner_2 2026-07-30 factor screen: batch candidate exploration.
Evaluates 12 candidate factor families on the 15-asset tradable universe,
validation window 2020-01-01..2026-07-30, admission horizon 10d.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   print_result, IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
macro = {
    "VIX": load_index("VIX"),
    "DXY": load_index("DXY"),
    "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"),
    "EURUSD": load_index("EURUSD"),
}
print(f"Panel dates {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows, {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library panels loaded: {list(lib.keys())}")


def ret_series(c, n=1):
    return c.pct_change(n)


def zscore(s, w):
    m = s.rolling(w).mean()
    sd = s.rolling(w).std()
    return (s - m) / sd


# ---------- candidate definitions ----------
def f_vol_ratio_20x60(c, v, o, h, l, m):
    v = v.replace(0, np.nan)
    r = v.rolling(20).mean() / v.rolling(60).mean()
    return r


def f_range_pos_20(c, v, o, h, l, m):
    hi = h.rolling(20).max()
    lo = l.rolling(20).min()
    return (c - lo) / (hi - lo)


def f_acorr_10(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(60).apply(lambda x: x.iloc[:-1].corr(x.shift(1).iloc[1:]), raw=False)


def f_dxy_beta_60(c, v, o, h, l, m):
    dxy = m["DXY"].pct_change()
    r = c.pct_change()
    x = dxy.reindex(r.index)
    cov = r.rolling(60).cov(x)
    var = x.rolling(60).var()
    return cov / var


def f_crypto_beta_60(c, v, o, h, l, m):
    btc = m.get("BTC_close") if "BTC_close" in m else None
    r = c.pct_change()
    btc_r = close["BTC"].pct_change()
    cov = r.rolling(60).cov(btc_r)
    var = btc_r.rolling(60).var()
    return cov / var


def f_rate_beta_60(c, v, o, h, l, m):
    us10y_d = close["US10Y"].diff()
    r = c.pct_change()
    cov = r.rolling(60).cov(us10y_d)
    var = us10y_d.rolling(60).var()
    return cov / var


def f_downside_vol_ratio_20(c, v, o, h, l, m):
    r = c.pct_change()
    neg = r.clip(upper=0)
    ds = (neg ** 2).rolling(20).mean() ** 0.5
    sd = r.rolling(20).std()
    return ds / sd


def f_skew_20(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(20).skew()


def f_dist_60d_low(c, v, o, h, l, m):
    lo = l.rolling(60).min()
    return c / lo - 1.0


def f_ohlc_pos_20(c, v, o, h, l, m):
    eff = (c - o) / (h - l)
    return eff.rolling(20).mean()


def f_dd_60(c, v, o, h, l, m):
    hi = c.rolling(60).max()
    return c / hi - 1.0


def f_vol_ratio_20x60_ret(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(20).std() / r.rolling(60).std()


def f_mom_45d_skip5(c, v, o, h, l, m):
    return c.shift(5) / c.shift(50) - 1.0


def f_bollinger_pos_20(c, v, o, h, l, m):
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    return (c - ma) / sd


def f_high_low_range_20(c, v, o, h, l, m):
    return (h.rolling(20).max() - l.rolling(20).min()) / c


CANDIDATES = [
    ("vol_ratio_20x60", f_vol_ratio_20x60, "volume participation trend (20v/60v)"),
    ("range_pos_20", f_range_pos_20, "20d range position (close in 20d range)"),
    ("acorr_60_1", f_acorr_10, "60d lag-1 autocorrelation of daily returns"),
    ("dxy_beta_60", f_dxy_beta_60, "60d rolling beta to DXY returns"),
    ("crypto_beta_60", f_crypto_beta_60, "60d rolling beta to BTC returns"),
    ("rate_beta_60", f_rate_beta_60, "60d rolling beta to US10Y yield change"),
    ("downside_vol_ratio_20", f_downside_vol_ratio_20, "downside semideviation / total vol (20d)"),
    ("skew_20", f_skew_20, "20d return skewness"),
    ("dist_60d_low", f_dist_60d_low, "distance above 60d low"),
    ("ohlc_pos_20", f_ohlc_pos_20, "20d avg intraday close location (close-open)/(high-low)"),
    ("dd_60", f_dd_60, "60d max drawdown depth"),
    ("vol_ratio_20x60_ret", f_vol_ratio_20x60_ret, "return vol regime ratio 20/60"),
    ("mom_45d_skip5", f_mom_45d_skip5, "45d momentum skip 5 (novel window)"),
    ("bollinger_pos_20", f_bollinger_pos_20, "20d Bollinger z-score of close"),
    ("high_low_range_20", f_high_low_range_20, "20d high-low range relative to close"),
]

results = {}
for name, fn, desc in CANDIDATES:
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib, close), 4)
    results[name] = res
    print_result(f"{name} [{desc}]", res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")

print("\n\n===== SUMMARY =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:24s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} -> {'PASS' if ok else 'fail'}")
