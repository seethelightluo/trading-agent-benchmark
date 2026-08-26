"""miner_1 cycle 2033-01-06: sweep NEW candidate factor families.
Visible history up to 2033-01-05 (last completed trading day). No lookahead.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084 (10d horizon).
15-instrument cross-asset universe; >=8 valid instruments per date for IC obs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2033-01-05'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    closes, highs, lows, opens = {}, {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists():
            f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        closes[a] = df['close'].astype(float)
        highs[a] = df['high'].astype(float)
        lows[a] = df['low'].astype(float)
        opens[a] = df['open'].astype(float)
    return closes, highs, lows, opens

closes, highs, lows, opens = load(ASSETS, VISIBLE_END)
close = pd.DataFrame(closes).dropna()
high = pd.DataFrame(highs).reindex(close.index)
low = pd.DataFrame(lows).reindex(close.index)
open_ = pd.DataFrame(opens).reindex(close.index)
rets = close.pct_change().dropna()
fwd5  = rets.shift(-5).rolling(5).mean()
fwd10 = rets.shift(-10).rolling(10).mean()
fwd20 = rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

def mac(c):
    df = pd.read_csv(INDEX_DIR/f'{c}.csv', parse_dates=['date'])
    df = df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
    return df
vix = mac('VIX'); dxy = mac('DXY'); jpy = mac('USDJPY')

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
    fv_ = fv.reindex(fwd10.index)
    ic = compute_ic(fv, fwd10); ic5 = compute_ic(fv, fwd5); ic20 = compute_ic(fv, fwd20)
    print(f"{name}[10]: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"dates_ok={ic['dates_ok']} hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv_):.3f} "
          f"| [5]IC={ic5['IC']:.4f} [20]IC={ic20['IC']:.4f}", flush=True)
    return ic

# A) gold trend 60d cross-sectional (risk-on/off regime diff)
xau_mom = close['XAU'].pct_change(60)
report("A xau_trend60_xsec", pd.DataFrame({a: xau_mom for a in ASSETS}).reindex(fwd10.index))

# B) correlation of each asset 10d returns to XAU 10d over 60d (gold linkage)
xau10 = close.pct_change(10)['XAU']
corr_xau = pd.DataFrame({a: close.pct_change(10)[a].rolling(60).corr(xau10) for a in ASSETS}).reindex(fwd10.index)
report("B corr_to_gold_60", corr_xau)

# C) overnight/gap avg 30d
gap = (open_/close.shift(1)-1)
report("C overnight_avg30", gap.rolling(30).mean().reindex(fwd10.index))

# D) amplitude ratio vol regime 10/60
amp = (high-low)/close
volreg = amp.rolling(10).mean().div(amp.rolling(60).mean())
report("D amp_ratio_10_60", volreg.reindex(fwd10.index))

# E) close location in 20d range (where in range close sits) avg
loc = (close-low.rolling(20).min())/(high.rolling(20).max()-low.rolling(20).min())
report("E close_loc_20d", loc.reindex(fwd10.index))

# F) drawdown from running 60d high, persisted-like candidate
dd = (close/close.rolling(60).max()-1)
report("F retrace_high_60", dd.reindex(fwd10.index))

# G) yield curve steepness proxy: US10Y - CN10Y momentum (cross-asset carry regime)
spread = close['US10Y'] - close['CN10Y']
report("G us10_cn10_spread_xsec", pd.DataFrame({a: spread for a in ASSETS}).reindex(fwd10.index))

# H) VIX 20d return cross-applied
vix_r = vix.pct_change(20)
report("H vix_roc20_xsec", pd.DataFrame({a: vix_r for a in ASSETS}).reindex(fwd10.index))

# I) USDJPY 10d return cross-section (carry/fx risk regime)
jpy_r = jpy.pct_change(10)