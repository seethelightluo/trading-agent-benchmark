"""miner_1 2034-08-25: exploration screen for NEW factor families (not in library).
Candidates tested (all computable from OHLC + macro, no volume dependency):
 A bollz_20d   : Bollinger z-score reversal (close-SMA20)/std20, dir -
 B ddepth_60d  : drawdown depth 1 - close/rolling_max(close,60), dir + (deep-dd rebound)
 C dskew_60d   : downside-risk share sum(min(r,0)^2)/sum(r^2) over 60d
 D range_vol20 : mean((high-low)/close, 20d) vol-level factor
 E gap_rev_1d  : overnight gap reversal -(open/prev_close - 1)
 F xs_rev_5d   : cross-sectional relative 5d reversal -(ret5 - mean(ret5))
IC computed on 13 live assets (HSI/CN10Y flat artifacts excluded); full-15 also reported.
"""
import numpy as np
import pandas as pd

panel = pd.read_pickle('scripts/panel_cache_20340825.pkl')
close = panel['close']; opn = panel['open']; high = panel['high']; low = panel['low']
macro = panel['macro']; ret = panel['ret']

LIVE = [c for c in close.columns if c not in ('HSI', 'CN10Y')]

def fwd_ret(close, h):
    return close.shift(-h) / close - 1.0

def daily_rank_ic(signal, fwd, cols, min_n=8):
    ics, dates = [], []
    idx = signal.index.intersection(fwd.index)
    for t in idx:
        s = signal.loc[t, cols]; f = fwd.loc[t, cols]
        m = s.notna() & f.notna()
        if m.sum() < min_n:
            continue
        ic = s[m].rank().corr(f[m].rank())
        if np.isfinite(ic):
            ics.append(ic); dates.append(t)
    return np.array(ics), np.array(dates)

def ev(signal, close, cols, horizons=(1, 5, 10), start=None):
    if start is not None:
        signal = signal[signal.index >= start]
    out = {}
    for h in horizons:
        fwd = fwd_ret(close, h)
        ics, dates = daily_rank_ic(signal, fwd, cols)
        if len(ics) == 0:
            out[h] = (np.nan, np.nan, np.nan, 0); continue
        ic = float(np.mean(ics)); sd = float(np.std(ics, ddof=1))
        icir = ic / sd if sd > 0 else np.nan
        out[h] = (ic, icir, float(np.mean(ics > 0)), len(ics))
    return out

def report(name, sig, cols, start=None):
    r = ev(sig, close, cols, start=start)
    def f(k):
        ic, icir, hit, n = r[k]
        return f"IC{k}={ic:.4f} ICIR{k}={icir:.3f} hit={hit:.2f} n={n}"
    cov = float(sig[cols].notna().mean().mean())
    print(f"{name:14s} cov={cov:.3f} | {f(1)} | {f(5)} | {f(10)}")

# ---------- build signals ----------
sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std()
bollz = (close - sma20) / std20
print("bollz_20d tail sample (last 3):"); print(bollz[LIVE].tail(3).round(3).to_string())

ddepth = 1.0 - close / close.rolling(60).max()

r2 = ret.clip(upper=0.0)**2
rsum = (ret**2).rolling(60).sum()
dskew = r2.rolling(60).sum() / rsum

range_vol = ((high - low) / close).rolling(20).mean()

gap = opn / close.shift(1) - 1.0

ret5 = close / close.shift(5) - 1.0
xs_rev5 = -(ret5 - ret5[LIVE].mean(axis=1))

full_start = pd.Timestamp('2021-01-01')   # warm-up 2020 excluded from IC sample
print("\n=== LIVE 13-asset IC (HSI/CN10Y excluded), sample 2021-01-01.. ===")
report("A bollz_20d", -bollz, LIVE, full_start)
report("B ddepth_60d", ddepth, LIVE, full_start)
report("C dskew_60d", -dskew, LIVE, full_start)
report("D range_vol20", -range_vol, LIVE, full_start)
report("E gap_rev_1d", -gap, LIVE, full_start)
report("F xs_rev_5d", xs_rev5, LIVE, full_start)

print("\n=== FULL 15-asset IC (incl flat artifacts), for reference ===")
report("A bollz_20d", -bollz, close.columns, full_start)
report("B ddepth_60d", ddepth, close.columns, full_start)
report("C dskew_60d", -dskew, close.columns, full_start)
report("D range_vol20", -range_vol, close.columns, full_start)
report("E gap_rev_1d", -gap, close.columns, full_start)
report("F xs_rev_5d", xs_rev5, close.columns, full_start)

# recent-window check (last 250d) for regime relevance
recent = pd.Timestamp('2033-10-01')
print("\n=== LIVE 13-asset IC, recent 2033-10-01.. ===")
report("A bollz_20d", -bollz, LIVE, recent)
report("B ddepth_60d", ddepth, LIVE, recent)
report("C dskew_60d", -dskew, LIVE, recent)
report("D range_vol20", -range_vol, LIVE, recent)
report("E gap_rev_1d", -gap, LIVE, recent)
report("F xs_rev_5d", xs_rev5, LIVE, recent)
