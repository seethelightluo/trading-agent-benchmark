"""miner_3 2033-10-13. Pooled cross-sectional rank correlation of fresh candidates vs existing lib."""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2033-10-12'
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
rets=close.pct_change().dropna()
lib={}
lib['mom_10']=close.shift(5)/close.shift(15)-1.0
lib['mom_120']=close.shift(5)/close.shift(125)-1.0
lib['kaufman']=(close.diff(20).abs()).div(close.diff().abs().rolling(20).sum())
lib['skew']=(rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3))
lib['bb']=rets.rolling(20).std()
lib['vol_z']=rets.rolling(20).std().rank(axis=1)
ac1=close.rolling(120).apply(lambda x:x.autocorr(1) if len(x.dropna())>=30 else np.nan,raw=False)
lib['ac1']=ac1
lib['kurt']=rets.rolling(20).kurt()
lib['rng_pos']=rets.rolling(20).apply(lambda x:(x>0).mean(),raw=True)
cand={}
cand['retrace_120']=(close/close.rolling(120).max()-1.0)
cand['retrace_60']=(close/close.rolling(60).max()-1.0)
cand['mom_accel_10x60']=((close.shift(5)/close.shift(15)-1.0)-(close.shift(5)/close.shift(65)-1.0))
cand['hlr_20']=(high-low)/close.rolling(20).mean()
cand['mom_240']=(close.shift(5)/close.shift(245)-1.0)
cand['price_vs_med120']=(close/close.rolling(120).median()-1.0)
# also include macro factors in lib for correlation check
import os
for ff in ['beta_VIX_60','cny_beta_60','dxy_corr_change_20_60']:
    p='factors/'+ff+'.json'
    if os.path.exists(p):
        lib['lib:'+ff]=np.nan  # placeholder; real correlation needs artifact

for cn,fv in cand.items():
    fv=fv.reindex(close.index)
    out=[]
    for ln,lf in lib.items():
        if isinstance(lf,float):
            out.append((ln,np.nan)); continue
        lf=lf.reindex(close.index)
        m=fv.notna()&lf.notna()
        arr=fv.where(m).stack()
        arrl=lf.where(m).stack()
        # align by same index
        dfc=pd.DataFrame({'a':arr,'b':arrl}).dropna()
        if len(dfc)>1000:
            r=dfc['a'].rank().corr(dfc['b'].rank())
            out.append((ln,float(r)))
    rr=[r for _,r in out if r==r]
    mx=max(abs(r) for r in rr) if rr else np.nan
    print(f"{cn}: max_abs_lib_corr={mx:.3f}  |  "+", ".join(f"{ln}:{r:+.2f}" if r==r else f"{ln}:na" for ln,r in out))