"""miner3 2032-05-21: broad scan of candidate factor ideas (one-pass screen)."""
import pandas as pd, numpy as np
from scipy.stats import rankdata

panel = pd.read_pickle('scripts/panel_cache_20320521.pkl')
close = panel['close']; ret = panel['ret']; vol = panel['vol']; macro = panel['macro']
opn = panel['open']; hi = panel['high']; lo = panel['low']
print("panel:", close.shape, close.index.min().date(), "->", close.index.max().date(), flush=True)

def fwd_ret(h):
    return close.shift(-h) / close - 1.0

def row_rank(m):
    out = np.full(m.shape, np.nan)
    for i in range(m.shape[0]):
        row = m[i]
        msk = ~np.isnan(row)
        if msk.sum() > 0:
            out[i, msk] = rankdata(row[msk])
    return out

def ic_series(sig_rank, fwd_rank, min_n=8):
    ics, dates = [], []
    for i in range(sig_rank.shape[0]):
        s = sig_rank[i]; f = fwd_rank[i]
        m = ~(np.isnan(s) | np.isnan(f))
        if m.sum() < min_n:
            continue
        ic = np.corrcoef(s[m], f[m])[0, 1]
        if np.isfinite(ic):
            ics.append(ic); dates.append(i)
    return np.array(ics), dates

def eval_factor(signal, close_sub, horizons=(1, 2, 3, 5, 10), min_n=8):
    out = {}
    sig_rank = row_rank(signal.values)
    for h in horizons:
        fwd = fwd_ret(h).reindex(signal.index)
        fr = row_rank(fwd.values)
        ics, dates = ic_series(sig_rank, fr, min_n)
        if len(ics) == 0:
            out[h] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
        else:
            ic = float(np.mean(ics)); sd = float(np.std(ics, ddof=1))
            out[h] = dict(ic=ic, icir=ic/sd if sd > 0 else np.nan,
                          hit=float(np.mean(ics > 0)), n=len(ics))
    out['coverage'] = float(signal.notna().mean().mean())
    out['n_dates'] = int(signal.shape[0])
    return out

lnc = np.log(close)
sig = {}

# 1. Kaufman efficiency ratio (trend consistency): |c - c_n| / sum(|ret|)
for n in [10, 20, 40]:
    er = (lnc - lnc.shift(n)).abs() / (ret.abs().rolling(n).sum().replace(0, np.nan))
    sig[f'kaufman_er_{n}'] = er

# 2. Range position (close-location-value in recent range)
for n in [10, 20]:
    hi_n = hi.rolling(n).max(); lo_n = lo.rolling(n).min()
    clv = (close - lo_n) / (hi_n - lo_n).replace(0, np.nan)
    sig[f'clv_{n}d'] = clv

# 3. Rolling return skewness (distribution asymmetry)
for n in [20, 60]:
    sig[f'skew_{n}d'] = ret.rolling(n).skew()

# 4. Downside-vol ratio: downside std / total std (asymmetry of risk)
for n in [20, 60]:
    neg = ret.clip(upper=0)
    ds = neg.rolling(n).std()
    tot = ret.rolling(n).std()
    sig[f'downside_vol_ratio_{n}'] = ds / tot.replace(0, np.nan)

# 5. Vol-scaled reversal: 5d reversal normalized by 20d vol
rv20 = ret.rolling(20).std()
sig['rev5_volscaled'] = -(lnc - lnc.shift(5)) / rv20.replace(0, np.nan)

# 6. Open-to-close gap persistence (gap = open vs prev close)
gap = lnc - np.log(opn)
sig['gap_persist_5d'] = gap.rolling(5).mean()

# 7. Volume divergence: price up while volume contracts (weak rally)
vma20 = vol.rolling(20).mean()
sig['price_vol_div'] = np.sign(ret.rolling(5).mean()) * (vol / vma20.replace(0, np.nan) - 1.0)

# 8. Cross-asset relative momentum: asset 20d ret minus cross-sectional median 20d ret
mom20 = lnc - lnc.shift(20)
xs_med = mom20.median(axis=1).rolling(20, min_periods=1).median()
sig['rel_mom_20d'] = mom20.sub(xs_med, axis=0)

# 9. Drawdown depth (contrarian on deep drawdowns)
run_hi = close.cummax()
sig['dd_20d'] = close / run_hi - 1.0

# 10. Trend slope R2 (linear fit quality over window) - proxy via correlation of price with time
def trend_r2(px, n):
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    t = np.arange(n)
    for c in px.columns:
        s = px[c]
        out[c] = s.rolling(n).apply(lambda x: np.corrcoef(x, t)[0, 1]**2, raw=True)
    return out
sig['trend_r2_20d'] = trend_r2(close, 20)

# 11. VIX-relative return: asset 10d return minus VIX-scaled threshold (risk-off neutral)
vix = macro['VIX']
vix_ret10 = np.log(vix).diff(10)
sig['vix_regime_neutral_10d'] = (lnc - lnc.shift(10)) - 0.5 * vix_ret10.to_frame().mean(axis=1).values[:, None] if False else None

# 12. MA crossover gap: (ma5 - ma20)/ma20
ma5 = close.rolling(5).mean(); ma20 = close.rolling(20).mean()
sig['ma_cross_5x20'] = (ma5 - ma20) / ma20.replace(0, np.nan)

# 13. 60d skewness of log-returns (fat left tail warning)
sig['logskew_60d'] = np.log(1 + ret).rolling(60).skew()

sig = {k: v for k, v in sig.items() if v is not None}

results = {}
for name, s in sig.items():
    s = s.reindex(close.index)
    res = {}
    for label, sl in [('full', slice(None)), ('recent_2y', close.index[-520:]), ('recent_1y', close.index[-260:])]:
        res[label] = eval_factor(s.loc[sl], close.loc[sl])
    results[name] = res

print(f"{'factor':24s} {'FULL ic1':>9s} {'icir1':>7s} {'hit':>5s} | {'2Y ic1':>7s} {'icir1':>7s} | {'1Y ic1':>7s} {'icir1':>7s} | {'cov':>5s}", flush=True)
for name, res in results.items():
    h1 = res['full'].get(1, {}); r2 = res['recent_2y'].get(1, {}); r1 = res['recent_1y'].get(1, {})
    print(f"{name:24s} {h1.get('ic',np.nan):9.4f} {h1.get('icir',np.nan):7.3f} {h1.get('hit',np.nan):5.3f} | "
          f"{r2.get('ic',np.nan):7.4f} {r2.get('icir',np.nan):7.3f} | "
          f"{r1.get('ic',np.nan):7.4f} {r1.get('icir',np.nan):7.3f} | {res['full'].get('coverage',np.nan):5.3f}", flush=True)

import json
with open('scripts/miner3_scan_20320521.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner3_scan_20320521.json")
