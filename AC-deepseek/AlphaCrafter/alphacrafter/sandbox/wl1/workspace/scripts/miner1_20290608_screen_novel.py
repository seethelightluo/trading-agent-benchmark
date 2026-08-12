"""miner1 2029-06-08: explore novel factor ideas through 2029-06-07.

Candidates (single idea per script requirement: this script is a screening
batch; each factor is a distinct idea and will be validated individually).
Ideas target known pain: momentum whipsaw at regime turns, ETH-specific drag,
volatility-regime confusion.
"""
import pandas as pd, numpy as np

panel = pd.read_pickle('scripts/panel_cache.pkl')
C = panel['close']; O = panel['open']; H = panel['high']; L = panel['low']
R = panel['ret']; M = panel['macro']
gate_ic, gate_icir = 0.0070, 0.0840

factors = {}
# --- 1. Volatility-scaled 60d momentum (risk-adjusted trend) ---
f = (C.shift(5) / C.shift(65) - 1.0)
vol = R.rolling(60).std()
factors['vomom_60_skip5'] = f / vol

# --- 2. Drawdown factor: negative distance from 20d high (recovery/mean-rev) ---
factors['dd_20'] = C / C.rolling(20).max() - 1.0

# --- 3. Trend quality: fraction of positive days over 60d ---
pos = (R > 0).astype(float).rolling(60).mean()
factors['trend_qual_60'] = pos

# --- 4. Downside/upside vol ratio over 60d ---
def du_ratio(x):
    up = x[x > 0].var(ddof=0)
    dn = x[x < 0].var(ddof=0)
    return dn / up if up > 0 else np.nan
factors['down_up_vol_60'] = R.rolling(60).apply(du_ratio, raw=True)

# --- 5. Cross-sectional dispersion z: 5d return vs cross-sectional mean ---
r5 = C.pct_change(5)
cs_mean = r5.mean(axis=1)
cs_std = r5.std(axis=1)
factors['cs_disp_5'] = (r5 - cs_mean.to_frame(r5.index)) / cs_std.to_frame(r5.index)

# --- 6. Momentum acceleration: 20d minus 120d momentum (slope of trend) ---
m20 = C.shift(5) / C.shift(25) - 1.0
m120 = C.shift(5) / C.shift(125) - 1.0
factors['mom_accel_20_120'] = m20 - m120

# --- 7. High-touch persistence: mean of (C-L)/(H-L) over 20d (close in top of range) ---
rng = (H - L).replace(0, np.nan)
clv = ((C - L) / rng).rolling(20).mean()
factors['high_touch_20'] = clv

# --- 8. Skewness of 60d returns (crash-risk asymmetry) ---
factors['skew_60'] = R.rolling(60).skew()

# --- 9. Max drawdown over 60d ---
def maxdd(x):
    return (x / np.maximum.accumulate(x) - 1.0).min()
factors['maxdd_60'] = C.rolling(60).apply(lambda x: maxdd(x.values), raw=False)

# --- 10. VIX-regime gated momentum (momentum only when VIX falling) ---
vix = M['VIX']
vix_down = (vix < vix.rolling(20).mean()).astype(float)
factors['vixgate_mom_60'] = f * vix_down.reindex(f.index).ffill()

# --- 11. Range compression breakout: low vol-of-vol but strong trend ---
vov = R.rolling(20).std().rolling(60).std()
factors['vov_mom_60'] = f / vov

# --- 12. 10d momentum with skip (short-term trend, previously deprecated; re-test) ---
factors['mom_10d_skip5'] = C.shift(5) / C.shift(15) - 1.0


def ic_series_vec(X, F):
    valid = X.notna() & F.notna()
    n = valid.sum(axis=1)
    keep = n >= 8
    Xr = X.rank(axis=1).where(valid)
    Fr = F.rank(axis=1).where(valid)
    dX = Xr.sub(Xr.mean(axis=1), axis=0)
    dF = Fr.sub(Fr.mean(axis=1), axis=0)
    num = (dX * dF).sum(axis=1)
    den = np.sqrt((dX ** 2).sum(axis=1) * (dF ** 2).sum(axis=1))
    ic = (num / den).where(den > 0)
    ic = ic[keep & ic.notna()]
    return ic


def stats(ic_ser):
    if len(ic_ser) == 0:
        return dict(n=0, ic=float('nan'), icir=float('nan'), hit=float('nan'))
    return dict(n=len(ic_ser), ic=float(ic_ser.mean()),
                icir=float(ic_ser.mean() / ic_ser.std()) if len(ic_ser) > 1 else float('nan'),
                hit=float((ic_ser > 0).mean()))


print(f"{'factor':22s} {'h1 ic':>9s} {'icir':>7s} {'hit':>6s} {'n':>5s} | {'365d ic':>9s} {'icir':>7s} {'n':>5s} | {'120d ic':>9s} {'icir':>7s} {'n':>5s} | gate(h1)")
results = {}
for name, f in factors.items():
    fwd = C.shift(-1) / C - 1.0
    ic1 = ic_series_vec(f, fwd)
    st = stats(ic1)
    last = ic1.index.max()
    last365 = ic1[ic1.index >= last - pd.Timedelta(days=365)]
    last120 = ic1[ic1.index >= last - pd.Timedelta(days=120)]
    s365 = stats(last365); s120 = stats(last120)
    ok = (abs(st['ic']) >= gate_ic) and (abs(st['icir']) >= gate_icir)
    results[name] = dict(full=st, last365=s365, last120=s120)
    print(f"{name:22s} {st['ic']:+9.5f} {st['icir']:+7.4f} {st['hit']:6.3f} {st['n']:5d} | "
          f"{s365['ic']:+9.5f} {s365['icir']:+7.4f} {s365['n']:5d} | {s120['ic']:+9.5f} {s120['icir']:+7.4f} {s120['n']:5d} | {ok}")

import json
json.dump({k: {kk: {kkk: (float(vvv) if isinstance(vvv, (np.floating, float)) else vvv)
                  for kkk, vvv in vv.items()} for kk, vv in v.items()}
           for k, v in results.items()},
          open('scripts/miner1_20290608_screen_novel.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner1_20290608_screen_novel.json")
