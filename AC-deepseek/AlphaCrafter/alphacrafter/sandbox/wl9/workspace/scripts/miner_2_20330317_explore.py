"""miner_2 cycle 2033-03-17: explore novel factors + revalidate effective library.
Visible through 2033-03-16. No lookahead. Gates: abs daily paper IC >=0.0070,
abs ICIR>=0.084 (10d horizon) over the 15-asset tradable universe.
Warm-up research only (no account/backtest/step actions).
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2033-03-16'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    closes, highs, lows, vols = {}, {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists(): f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        closes[a] = df['close'].astype(float); highs[a] = df['high'].astype(float)
        lows[a] = df['low'].astype(float)
        vols[a] = df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan, index=df.index)
    return closes, highs, lows, vols

closes, highs, lows, vols = load(ASSETS, VISIBLE_END)
close = pd.DataFrame(closes).dropna()
high = pd.DataFrame(highs).reindex(close.index); low = pd.DataFrame(lows).reindex(close.index)
vol = pd.DataFrame(vols).reindex(close.index)
rets = close.pct_change().dropna()
fwd5 = rets.shift(-5).rolling(5).mean()
fwd10 = rets.shift(-10).rolling(10).mean()
fwd20 = rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

def mac(c):
    df = pd.read_csv(INDEX_DIR / f'{c}.csv', parse_dates=['date'])
    df = df[df['date'] <= VISIBLE_END].set_index('date')['close'].astype(float); return df
vix = mac('VIX'); dxy = mac('DXY'); usdcny = mac('USDCNY')
dVIX = vix.pct_change(); dCNY = usdcny.pct_change(); dDXY = dxy.pct_change()

def compute_ic(fv, fwd, min_dates=30, start=None):
    f = fv.reindex(fwd.index); idx = fwd.index
    if start: idx = idx[idx >= pd.Timestamp(start)]
    ics = []; ok = 0
    for d in idx:
        x = f.loc[d]; y = fwd.loc[d]; m = x.notna() & y.notna()
        if m.sum() >= 8:
            ok += 1; a = x[m].rank().values; b = y[m].rank().values
            if np.std(a) > 0 and np.std(b) > 0: ics.append(np.corrcoef(a, b)[0, 1])
    ics = np.array(ics)
    if len(ics) < min_dates: return {'IC': 0., 'ICIR': 0., 'n': len(ics), 'hit': 0., 'cov': 0., 'ok': ok}
    hit = float((ics > 0).mean()); cov = float(f.notna().mean().mean())
    mu = ics.mean(); sd = ics.std(); icir = mu / sd * np.sqrt(len(ics)) if sd > 0 else 0
    return {'ic': float(mu), 'icir': float(icir), 'n': len(ics), 'hit': hit, 'cov': cov, 'ok': ok}

def turnover(fv):
    fv = fv.dropna(how='all')
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1] / 2)).fillna(0)
    return float((s.diff() != 0).mean().mean())

def report(name, fv, start=None):
    if fv is None: return None
    a = compute_ic(fv, fwd10, start=start); b = compute_ic(fv, fwd5, start=start)
    c = compute_ic(fv, fwd20, start=start)
    pass10 = abs(a['ic']) >= 0.0070 and abs(a['icir']) >= 0.084
    print(f"[{'OK' if pass10 else '--'}] {name}: IC={a['ic']:.4f} ICIR={a['icir']:.4f} "
          f"n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} "
          f"| [5]{b['ic']:.4f}/{b['icir']:.2f} [20]{c['ic']:.4f}/{c['icir']:.2f}", flush=True)
    return a

# =================== LIBRARY PROXIES for correlation audit ===================
lib = {}
lib['ac1_120d'] = rets.rolling(120, min_periods=60).apply(lambda x: np.corrcoef(x[1:], x[:-1])[0, 1] if len(x) > 3 else np.nan, raw=False)
lib['bb_width_20d'] = rets.rolling(20).std()
lib['beta_VIX_60'] = (rets.rolling(60).cov(dVIX)).div(dVIX.rolling(60).var())
lib['cny_beta_60'] = (rets.rolling(60).cov(dCNY)).div(dCNY.rolling(60).var())
dxyr = dDXY.reindex(close.index)
lib['dxy_corr_change_20_60'] = rets.rolling(20).corr(dxyr) - rets.rolling(60).corr(dxyr)
lib['kaufman_eff_20d'] = (close.diff(20).abs()).div(close.diff().abs().rolling(20).sum())
lib['mom_10d_skip5'] = close.shift(5) / close.shift(15) - 1.0
lib['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
lib['skew_20d'] = (rets - rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3))
lib['vol_z_20d'] = rets.rolling(20).std().rank(axis=1)

def libcorr(fv):
    """max |spearman| between candidate (cross-section pooled) and library factors."""
    cand = fv.stack().rename('cand')
    best = 0.0
    for name, lf in lib.items():
        other = lf.stack().rename('other')
        j = pd.concat([cand, other], axis=1).dropna()
        if len(j) < 100: continue
        # detrend by date (cross-section demeaned ranks) to avoid level correlation
        cand_c = cand - cand.groupby(level=0).transform('mean')
        cj = pd.concat([cand_c, lf.stack()], axis=1).dropna()
        if len(cj) < 100: continue
        r = np.corrcoef(cj.iloc[:, 0], cj.iloc[:, 1])[0, 1]
        best = max(best, abs(r))
    return float(best)

print("\n===== RERUN EFFECTIVE LIBRARY (10d, full) =====")
for k, v in lib.items(): report(k, v)
print("\n===== RERUN RECENT 2Y (drift check) =====")
RECENT = '2031-01-01'
for k, v in lib.items(): report(f"{k}[r]", v, s