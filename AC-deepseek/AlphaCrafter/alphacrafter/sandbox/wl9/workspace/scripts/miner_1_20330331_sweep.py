"""miner_1 cycle 2033-03-31: sweep NEW candidate factor families.
Visible history up to 2033-03-30 (last completed trading day). No lookahead.
Admission gates (15-asset cross-asset benchmark): abs daily paper IC >= 0.0070,
abs ICIR >= 0.084 (10d horizon). >=8 valid instruments per date for IC obs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2033-03-30'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    closes, highs, lows, opens, vols = {}, {}, {}, {}, {}
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
        if 'volume' in df.columns:
            vols[a] = df['volume'].astype(float)
    return closes, highs, lows, opens, vols

closes, highs, lows, opens, vols = load(ASSETS, VISIBLE_END)
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
    return df[~df.index.duplicated(keep='last')]
vix = mac('VIX'); dxy = mac('DXY'); jpy = mac('USDJPY'); eur = mac('EURUSD')

def compute_ic(fv, fwd, min_dates=30, start=None):
    fv = fv.reindex(fwd.index)
    mask = np.ones(len(fwd.index), dtype=bool)
    if start is not None:
        mask = fwd.index >= pd.Timestamp(start)
    ics = []; n_ok = 0
    for d in fwd.index[mask]:
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
    ic_r = compute_ic(fv, fwd10, start='2030-01-01')
    print(f"{name}[10]: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"dates_ok={ic['dates_ok']} hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f} "
          f"| [5]IC={ic5['IC']:.4f} [20]IC={ic20['IC']:.4f} | recent30+ {ic_r['IC']:.4f}/{ic_r['ICIR']:.4f}", flush=True)
    return ic

# A) TREND_EFF_60: Kaufman efficiency ratio 60d (price path efficiency)
eff = close.diff().abs().rolling(60).mean().div((close.pct_change().abs().rolling(60).sum()))
report("A trend_eff_60", eff)

# B) REVERSE mom 20d (short-term reversal)
report("B mom_20d_rev", close.shift(5)/close.shift(25)-1)

# C) RANGE / amplitude regime ratio 10/60 (contraction -> expansion)
amp = (high-low)/close
report("C amp_ratio_10_60", amp.rolling(10).mean().div(amp.rolling(60).mean()))

# D) CLOSE location in 20d range (scaled where close sits)
loc = (close-low.rolling(20).min())/(high.rolling(20).max()-low.rolling(20).min())
report("D close_loc_20d", loc)

# E) GAP/overnight avg 30d
gap = (open_/close.shift(1)-1)
report("E overnight_avg30", gap.rolling(30).mean())

# F) realized vol panel 20d (high vol -> ?)
rv = rets.rolling(20).std()
report("F vol_20d", rv)

# G) VIX level regime cross-applied (VIX 60d mean)
vix_m = vix.rolling(20).mean()
report("G vix_level20_xsec", pd.DataFrame({a: vix_m.reindex(close.index) for a in ASSETS}).reindex(fwd10.index))

# H) DXY 10d momentum cross-applied (dollar regime)
dxy_r = dxy.pct_change(10)
report("H dxy_roc10_xsec", pd.DataFrame({a: dxy_r.reindex(close.index) for a in ASSETS}).reindex(fwd10.index))

# I) yield-curve spread momentum xsec (US10Y - CN10Y change)
spread = (close['US10Y'] - close['CN10Y'])
report("I us10_cn10_spread_lvl", pd.DataFrame({a: spread for a in ASSETS}).reindex(fwd10.index))

# J) VIX RANK of recent 20d return across assets at high-vix days (conditional risk timing)
vix_hi = (vix > vix.rolling(120).quantile(0.7)).astype(float).reindex(fwd10.index)
mom = close.pct_change(20)
report("J mom20_vixhi_cond", (mom*vix_hi).reindex(fwd10.index))

# print best candidates for follow-up
print("\nDone sweep. Candidates with abs IC>=0.007 & abs ICIR>=0.084 in [10] identified above.", flush=True)
