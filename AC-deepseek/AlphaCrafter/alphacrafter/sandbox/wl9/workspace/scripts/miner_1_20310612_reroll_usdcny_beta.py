"""miner_1 cycle 2031-06-12: re-roll USDCNY beta factor (macro risk-off tilt).

Motivation: ensemble heavily weights beta_VIX_60(neg), cny_beta_60, dxy_corr_change.
RMB depreciation pressure (USDCNY rising) is a classic risk-off/funding-stress
signal; CNY-beta should predict NEGATIVE forward returns (like VIX beta).
Existing cny_beta_60 in library has direction +1; test an explicit beta on
USDCNY returns with 60d window across regimes including 2031 drawdowns.
"""
import json
import base64
import zlib
from io import StringIO
from pathlib import Path
import pandas as pd

STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF = pd.Timestamp('2031-06-11')  # completed bars only

def load_asset(a):
    f = STOCK_DIR/f'{a}.csv'
    if not f.exists(): f = INDEX_DIR/f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
    return df[df['date'] <= CUTOFF].set_index('date')

def load_macro(m):
    df = pd.read_csv(INDEX_DIR/f'{m}.csv', parse_dates=['date']).sort_values('date')
    return df[df['date'] <= CUTOFF].set_index('date')

closes = {a: load_asset(a)['close'].astype(float) for a in ASSETS}
usdcny = load_macro('USDCNY')['close'].astype(float)

ret = pd.DataFrame({a: c.pct_change() for a, c in closes.items()}).dropna(how='all')
dcny = usdcny.pct_change().rename('usdcny')
ret = ret.join(dcny, how='left')

def beta60(x, y):
    z = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna().tail(60)
    if len(z) < 30: return None
    var = float(z['y'].var())
    if var < 1e-14: return None
    return float(z['x'].cov(z['y'])/var)

def ic_series(factor, fwd, min_assets=8):
    """Daily cross-sectional Spearman IC of factor vs forward return."""
    rows = {}
    dates = factor.dropna(how='all').index
    for t in dates:
        fv = factor.loc[t]
        fr = fwd.loc[t]
        pair = pd.concat([fv.rename('f'), fr.rename('r')], axis=1).dropna()
        if len(pair) < min_assets: continue
        rows[t] = pair['f'].corr(pair['r'], method='spearman')
    return pd.Series(rows)

def report(tag, factor, fwd, decay_horizons=(1,2,3,5,10,20), direction=None):
    out = {}
    for h in decay_horizons:
        fr = fwd if h == 10 else ret[ASSETS].shift(-h)
        ic = ic_series(factor, fr)
        out[h] = ic
    # summary at 10d horizon
    ic10 = out[10]
    n = len(ic10.dropna())
    mean = float(ic10.mean()) if n else float('nan')
    std = float(ic10.std()) if n else float('nan')
    icir = mean/std if std > 0 else float('nan')
    hit = float((ic10 > 0).mean()) if n else float('nan')
    # turnover: rank-IC of day-over-day factor change at 10d rebalance spacing
    fac_10 = factor.resample('10B').last()
    turn = float(fac_10.diff().abs().mean().mean()) if len(fac_10) > 2 else float('nan')
    cov = float(factor.notna().sum().sum()) / (factor.shape[0]*factor.shape[1])
    valid_dates = int(ic10.notna().sum())
    print(f'--- {tag} ---')
    print(f'  IC(10d) mean={mean:+.4f} icir={icir:+.4f} hit={hit:.3f} n_dates={valid_dates} cov={cov:.3f} turnover={turn:.4f}')
    if direction == -1:
        print(f'  signed(-1): IC={-mean:+.4f} ICIR={-icir:+.4f} hit(-1)={1-hit:.3f}')
    print('  decay IC:', {h: round(float(s.mean()), 4) for h, s in out.items()})
    return out

print(f'ASSETS={len(ASSETS)} dates_visible={CUTOFF.date()}')

# === Candidate A: cny_beta_60 (raw sign, then report signed -1) ===
fac = pd.DataFrame(index=ret.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    fac[a] = beta60(ret[a], ret['usdcny'])
fwd10 = ret[ASSETS].shift(-10)
report('cny_beta_60 raw(positive direction)', fac, fwd10, direction=-1)

# === Candidate B: cny_beta_20 (faster beta) ===
def beta20(x, y):
    z = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna().tail(20)
    if len(z) < 12: return None
    var = float(z['y'].var())
    if var < 1e-14: return None
    return float(z['x'].cov(z['y'])/var)
fac20 = pd.DataFrame(index=ret.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    fac20[a] = beta20(ret[a], ret['usdcny'])
report('cny_beta_20 raw', fac20, fwd10, direction=-1)

# === Candidate C: interaction cnyxVIX (USDCNY vol regime) ===
facC = pd.DataFrame(index=ret.index, columns=ASSETS, dtype=float)
vix = load_macro('VIX')['close'].astype(float).pct_change()
dcny_abs = ret['usdcny'].abs()
regime = dcny_abs.rolling(20).mean().rank(pct=True)
dabs = dcny_abs.rolling(20).mean()
facC = fac.mul(dabs, axis=0)
report('cny_beta_60 * |dCNY|20', facC, fwd10, direction=-1)

# regime split: IC in high/low USDCNY-stress periods
stress = dabs.rolling(20).mean()
med = stress.median()
ic10 = ic_series(fac, fwd10)
hi = ic10[stress.reindex(ic10.index) >= med]
lo = ic10[stress.reindex(ic10.index) < med]
print(f'--- regime split cny_beta_60 ---')
print(f'  high-stress IC mean={hi.mean():+.4f} n={len(hi)} | low-stress IC mean={lo.mean():+.4f} n={len(lo)}')

# coverage of factor values
print(f'cny_beta_60 valid cells={int(fac.notna().sum().sum())} / {fac.shape[0]*fac.shape[1]}')