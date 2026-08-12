import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=2600)
    if d is None or len(d)<120: d=get_index_daily_data(s, days=2600)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index()
# simple, interpretable residual 5d reversal, vol scaled, lagged one day
r5=px.pct_change(5)
med=r5.median(axis=1)
vol=px.pct_change().rolling(60,min_periods=40).std()
sig=-(r5.sub(med,axis=0)).div(vol)
sig=sig.shift(1)
fwd=px.shift(-5).div(px)-1
rows=[]
for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((dt,ic,len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# evaluate full and recent regimes
for label,q in [('full',res),('recent_2028',res[res.index>='2028-01-01']),('2029plus',res[res.index>='2029-01-01'])]:
    print(label,'obs',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
# rank turnover and coverage
valid=sig.notna().sum(axis=1)/len(U)
print('coverage',round(valid.mean(),4),'turnover',round((sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
os.makedirs('scripts',exist_ok=True)
out='scripts/miner_3_20300404_residual_simple5_signal.csv'
sig.to_csv(out,index_label='date'); print('artifact',out,'dates',len(px),'instruments',len(frames))
