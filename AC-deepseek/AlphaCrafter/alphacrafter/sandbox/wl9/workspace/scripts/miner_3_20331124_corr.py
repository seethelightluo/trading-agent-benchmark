"""miner_3 2033-11-24: correlation of fresh passing candidates vs effective library.
Compute max_abs_library_correlation (recent regime 2032+).
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-11-23'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(assets,end):
    C,H,L={},{},{}
    for a in assets:
        f=SD/f'{a}.csv'
        if not f.exists(): f=ID/f'{a}.csv'
        df=pd.read_csv(f,parse_dates=['date']); df=df[df['date']<=end].sort_values('date').set_index('date')
        C[a]=df['close'].astype(float); H[a]=df['high'].astype(float); L[a]=df['low'].astype(float)
    return C,H,L
closes,highs,lows=load(ASSETS,VISIBLE_END)
close=pd.DataFrame(closes).dropna(); high=pd.DataFrame(highs).reindex(close.index); low=pd.DataFrame(lows).reindex(close.index)
r=close.pct_change().dropna()
RUN='2032-01-01'

lib={}
lib['mom_10d_skip5']=close.shift(5)/close.shift(15)-1.0
lib['mom_120d_skip5']=close.shift(5)/close.shift(125)-1.0
lib['kaufman_eff_20d']=(close.diff(20).abs()).div(close.diff().abs().rolling(20).sum())
lib['skew_20d']=(r-r.rolling(20).mean()).pow(3).rolling(20).mean().div(r.rolling(20).std().pow(3))
lib['bb_width_20d']=r.rolling(20).std()
lib['vol_z_20d']=r.rolling(20).std().rank(axis=1)
lib['ac1_120d']=close.rolling(120).apply(lambda x: x.autocorr(1) if len(x.dropna())>=30 else np.nan, raw=False)

rng=(high-low)/close
fresh={
 'below_high60': close/close.rolling(60).max()-1.0,
 'upper_wave_20': ((high-close)/close).rolling(20).mean(),
 'zscore_20':(close-close.rolling(20).mean())/close.rolling(20).std().replace(0,np.nan),
 'range_ratio_20_60':rng.rolling(20).mean()/rng.rolling(60).mean()-1.0,
 'kaufman_eff_60':(close.diff(60).abs()).div(close.diff().abs().rolling(60).sum()),
}
def rec(df): return df[df.index>=pd.Timestamp(RUN)]
def avgcorr(a,b):
    ra=a.rank(axis=1); rb=b.rank(axis=1); cs=[]
    for d in a.index:
        x=ra.loc[d]; y=rb.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8 and np.std(x[m])>0 and np.std(y[m])>0:
            cs.append(np.corrcoef(x[m],y[m])[0,1])
    return float(np.mean(cs)) if cs else 0
for name,f in fresh.items():
    a=rec(f).dropna(how='all')
    best=0; bestk=''
    for k,lf in lib.items():
        b=rec(lf).reindex(a.index)
        c=abs(avgcorr(a,b))
        if c>best: best=c; bestk=k
    print(f"{name}: max_abs_lib_corr={best:.3f} (vs {bestk})")