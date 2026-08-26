"""miner_3 cycle 2032-10-14: sweep NEW candidate factor families.
Visible history up to 2032-10-13 (last completed trading day). No lookahead.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084 (10d horizon).
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2032-10-13'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    closes, highs, lows = {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists():
            f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        closes[a] = df['close'].astype(float)
        highs[a] = df['high'].astype(float)
        lows[a] = df['low'].astype(float)
    return closes, highs, lows

closes, highs, lows = load(ASSETS, VISIBLE_END)
close = pd.DataFrame(closes).dropna()
high = pd.DataFrame(highs).reindex(close.index)
low = pd.DataFrame(lows).reindex(close.index)
rets = close.pct_change().dropna()
ret_idx = rets.index
fwd5  = rets.shift(-5).rolling(5).mean()
fwd10 = rets.shift(-10).rolling(10).mean()
fwd20 = rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

def mac(c):
    df = pd.read_csv(INDEX_DIR/f'{c}.csv', parse_dates=['date'])
    df = df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
    return df
dxy = mac('DXY'); vix = mac('VIX'); usdcny = mac('USDCNY')

def compute_ic(fv, fwd, min_dates=30):
    fv = fv.reindex(fwd.index)
    ics = []; n_ok = 0
    for d in fwd.index:
        f = fv.loc[d]; r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= 8:
            n_ok += 1
            fv_ = f[m].rank().values; rv_ = r[m].rank().values
            if fv_.std() > 0 and rv_.std() > 0:
                ics.append(np.corrcoef(fv_, rv_)[0,1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return {'IC':0.0,'ICIR':0.0,'n':len(ics),'hit':0.0,'cov':0.0,'dates_ok':n_ok}
    hit = float((ics>0).mean()); cov = float(fv.notna().mean().mean())
    mu=ics.mean(); sd=ics.std(); icir=mu/sd*np.sqrt(len(ics)) if sd>0 else 0.0
    return {'IC':float(mu),'ICIR':float(icir),'n':len(ics),'hit':hit,'cov':cov,'dates_ok':n_ok}

def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv):
    ic = compute_ic(fv, fwd10); ic5 = compute_ic(fv, fwd5); ic20 = compute_ic(fv, fwd20)
    fv = fv.reindex(fwd10.index)
    print(f"{name}[10]: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"dates_ok={ic['dates_ok']} hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f} "
          f"| [5]IC={ic5['IC']:.4f} [20]IC={ic20['IC']:.4f}", flush=True)
    return ic

# A) reversal 2d/3d
report("A rev_2d", (-rets.rolling(2).mean()).reindex(fwd10.index))
report("A2 rev_3d", (-rets.rolling(3).mean()).reindex(fwd10.index))
# B) down/up vol asymmetry
retp = rets.where(rets>0, 0.0); retn = rets.where(rets<0, 0.0)
vol_up = pd.DataFrame({a: retp[a].rolling(20).std() for a in ASSETS}).reindex(fwd10.index)
vol_dn = pd.DataFrame({a: abs(retn[a]).rolling(20).std() for a in ASSETS}).reindex(fwd10.index)
report("B downup_vol_asym20", (vol_dn - vol_up))
# C) range pos 10d
rngpos = ((close - low)/(high-low).replace(0,np.nan)).rolling(10).mean()
report("C range_pos10", rngpos.reindex(fwd10.index))
# D) wti/xau relmom
wti_xau = close['WTI']/close['XAU']
D = pd.DataFrame({a: wti_xau.pct_change(20) for a in ASSETS}).reindex(fwd10.index)
report("D wti_xau_relmom20", D)
# E) xau/spx rel
xau_spx = close['XAU']/close['SPX']
E = pd.DataFrame({a: xau_spx.pct_change(20) for a in ASSETS}).reindex(fwd10.index)
report("E xau_spx_relmom20", E)
# F) recovery from low60
lo60 = close.rolling(60).min(); hi60 = close.rolling(60).max()
recover = ((close - lo60)/(hi60 - lo60)).replace([np.inf,-np.inf],np.nan)
report("F recover_from_low60", recover.reindex(fwd10.index))
# G) vol term
vol_s = pd.DataFrame({a: rets[a].rolling(10).std() for a in ASSETS}).reindex(fwd10.index)
vol_l = pd.DataFrame({a: rets[a].rolling(40).std() for a in ASSETS}).reindex(fwd10.index)
report("G vol_10_40_diff", (vol_s - vol_l))
# H) mom3/vol10
mom3 = pd.DataFrame({a: close[a].pct_change(3) for a in ASSETS}).reindex(fwd10.index)
vol10 = pd.DataFrame({a: rets[a].rolling(10).std() for a in ASSETS}).reindex(fwd10.index)
report("H mom3_vol10", (mom3/vol10))
# I) vol of vol
vol20 = pd.DataFrame({a: rets[a].rolling(20).std() for a in ASSETS}).reindex(fwd10.index)
vov = pd.DataFrame({a: vol20[a].rolling(60).std() for a in ASSETS}).reindex(fwd10.index)
report("I vol_of_vol20x60", vov)
# J) vix roc applied cross-sectionally
vix_roc20 = (vix/vix.shift(20)-1)
J = pd.DataFrame({a: vix_roc20 for a in ASSETS}).reindex(fwd10.index)
report("J vix_roc20", J)
# K) cny trend
cny_roc = (usdcny.shift(5)/usdcny-1)  # usdcny rising=CNY weakening
K = pd.DataFrame({a: cny_roc for a in ASSETS}).
report("K cny_roc5", K)
skew120 = pd.DataFrame({a: rets[a].rolling(120).skew() for a in ASSETS}).reindex(fwd10.index)
report("L skew120", skew120)
crypto = (close['BTC'].pct_change(10)+close['ETH'].pct_change(10))/2
M = pd.DataFrame({a: crypto for a in ASSETS}).reindex(fwd10.index)
report("M crypto_mom10", M)
carry = close['US10Y'] - close['CN10Y']
N = pd.DataFrame({a: carry.pct_change(20) for a in ASSETS}).reindex(fwd10.index)
report("N us_cn10y_diff20", N)
