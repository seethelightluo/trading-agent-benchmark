"""miner1 2033-09-09: screen NEW candidate factor ideas (distinct from library)
through 2033-09-08. Vectorized cross-sectional IC on 15-asset panel.
Gate: |IC1| >= 0.0070, |ICIR1| >= 0.0840 (full-window daily paper metrics).
Also report variant excluding HSI/CN10Y flat-data artifacts.
"""
import pandas as pd, numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

panel = pd.read_pickle('scripts/panel_cache_20330908.pkl')
close = panel['close']; ret = panel['ret']; vol = panel['vol']
open_ = panel['open']; high = panel['high']; low = panel['low']
macro = panel['macro']
print(f"panel loaded {time.time()-t0:.1f}s", flush=True)

def ic_series_vec(signal, close_, hz=1):
    sig_r = signal.rank(axis=1, pct=True).values
    fwd = (np.log(close_.shift(-hz) / close_)).values
    dates = signal.index
    ics = []
    for t in range(len(signal) - hz):
        s = sig_r[t]; f = fwd[t]
        m = np.isfinite(s) & np.isfinite(f)
        if m.sum() >= 8:
            ic = np.corrcoef(s[m], f[m])[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
    return pd.Series(ics, index=dates[:len(ics)])

def stats(ics):
    if len(ics) == 0:
        return dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
    sd = float(ics.std(ddof=1))
    return dict(ic=float(ics.mean()), icir=float(ics.mean()/sd) if sd > 0 else np.nan,
                hit=float((ics > 0).mean()), n=len(ics))

# ---------------- candidate signals (all use info through t only) ----------------
S = {}
# 1. Kaufman efficiency ratio 20d (trend quality): |net move| / path length
S['er20'] = (close - close.shift(20)).abs() / ret.abs().rolling(20).sum()
# 2. Bollinger %B 20x2: position within mean/std bands
ma20 = close.rolling(20).mean(); sd20 = close.rolling(20).std()
S['pctb_20'] = (close - ma20) / (2 * sd20)
# 3. Drawdown depth 120d: distance below rolling max (contrarian oversold)
S['dd_120'] = close / close.rolling(120).max() - 1.0
# 4. RSI-14 (classic oscillator, raw: low RSI -> oversold)
def rsi(px, n=14):
    d = px.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
S['rsi_14'] = rsi(close, 14)
# 5. Amihud illiquidity 20d (log): mean(|ret|/volume)
amihud = (ret.abs() / vol.replace(0, np.nan)).rolling(20).mean()
S['amihud_20'] = np.log1p(amihud * 1e9)
# 6. Downside/upside vol ratio 20d (risk asymmetry)
neg = ret.clip(upper=0); pos = ret.clip(lower=0)
S['downup_20'] = neg.rolling(20).std() / pos.rolling(20).std().replace(0, np.nan)
# 7. Volume z-score 20d (volume surge)
vol_ma = vol.rolling(20).mean(); vol_sd = vol.rolling(20).std()
S['volz_20'] = (vol - vol_ma) / vol_sd.replace(0, np.nan)
# 8. Trend strength: (MA20/MA60 - 1) / vol20
ma60 = close.rolling(60).mean()
S['maspread_20x60'] = (ma20 / ma60 - 1.0) / (ret.rolling(20).std() * np.sqrt(252)).replace(0, np.nan)
# 9. Skewness 20d (return asymmetry)
S['skew_20'] = ret.rolling(20).skew()
# 10. Range ratio 20d mean: (high-low)/close (intraday vol proxy)
S['range_20'] = ((high - low) / close).rolling(20).mean()
# 11. Overnight gap reversal: -(open/prev_close - 1)
S['gap_rev_1d'] = -(open_ / close.shift(1) - 1.0)
# 12. 5d momentum of 20d vol (vol trend, distinct from vol_of_vol)
rv20 = ret.rolling(20).std()
S['volmom_5x20'] = np.log(rv20) - np.log(rv20.shift(5))

GATE_IC, GATE_ICIR = 0.0070, 0.0840
results = {}
print(f"\nGates: |IC1|>={GATE_IC} |ICIR1|>={GATE_ICIR}\n")
for name, s in S.items():
    s = s.reindex(close.index)
    res = {}
    for label, sl in [('full', slice(None)), ('recent_2y', close.index[-520:]), ('recent_1y', close.index[-260:])]:
        ss = s.loc[sl]
        res[label] = {h: stats(ic_series_vec(ss, close.loc[sl], h)) for h in [1, 2, 3, 5, 10]}
    # variant excluding flat-artifact HSI/CN10Y on full window h=1
    sub = s.drop(columns=['HSI', 'CN10Y'], errors='ignore').reindex(close.index)
    res['full_noHCN'] = {1: stats(ic_series_vec(sub, close.drop(columns=['HSI','CN10Y'], errors='ignore'), 1))}
    results[name] = res
    f1 = res['full'][1]; f5 = res['full'][5]; f10 = res['full'][10]
    r2 = res['recent_2y'][1]; r1 = res['recent_1y'][1]
    nH = res['full_noHCN'][1]
    cov = float(s.notna().mean().mean())
    rp = s.rank(axis=1, pct=True)
    turn = float(rp.diff().abs().mean().mean())
    gate = "PASS" if (abs(f1['ic']) >= GATE_IC and abs(f1['icir']) >= GATE_ICIR) else "fail"
    print(f"{name:14s} [{gate}] FULL ic1={f1['ic']:+.4f} icir1={f1['icir']:+.3f} hit={f1['hit']:.3f} n={f1['n']:4d} | "
          f"ic5={f5['ic']:+.4f} icir5={f5['icir']:+.3f} | ic10={f10['ic']:+.4f} icir10={f10['icir']:+.3f} | "
          f"2Y ic1={r2['ic']:+.4f}/{r2['icir']:+.2f} 1Y ic1={r1['ic']:+.4f}/{r1['icir']:+.2f} "
          f"noHCN ic1={nH['ic']:+.4f}/{nH['icir']:+.2f} | cov={cov:.3f} turn={turn:.3f}", flush=True)

with open('scripts/miner1_20330909_screen.json', 'w') as f:
    json.dump({k: {kk: {h: vv for h, vv in v.items()} for kk, v in vv.items()} for k, vv in results.items()},
              f, indent=1, default=str)
print(f"\nsaved scripts/miner1_20330909_screen.json in {time.time()-t0:.1f}s")
