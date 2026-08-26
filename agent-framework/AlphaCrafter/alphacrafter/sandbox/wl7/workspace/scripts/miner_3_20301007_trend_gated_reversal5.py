import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d)>100:
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close
        frames[s]=x
p=pd.concat(frames,axis=1).sort_index().ffill()
r=np.log(p).diff()
# candidate: lagged 5d reversal, volatility normalized, only when asset's 20d trend is positive
# positive trend gate seeks pullbacks in established trends; cross-sectional ranks preserve breadth
rv=r.rolling(20).std()*np.sqrt(20)
ret5=np.log(p/p.shift(5))
trend20=np.log(p/p.shift(20))
raw=(-ret5/rv).shift(1)
gate=(trend20.shift(1)>0).astype(float)
f=raw*gate
# rank within date, requiring >=8 and forward returns available
out=[]
for h in [1,5,10,20]:
    fr=np.log(p.shift(-h)/p)
    vals=[]; nins=[]
    for dt in p.index:
        a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); nins.append(len(z))
    q=pd.Series(vals).dropna(); ic=q.mean(); sd=q.std(ddof=1)
    print(f'H={h} dates={len(q)} avg_n={np.mean(nins):.2f} IC={ic:.8f} ICIR={ic/sd*np.sqrt(252):.8f} hit={(q>0).mean():.4f}')
# coverage and rank turnover
valid=f.notna().sum().sum()/(len(f)*len(U)); ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).dropna().mean()
print(f'coverage={valid:.6f} rank_turnover={turnover:.6f} rows={len(p)} instruments={len(frames)} last={p.index.max().date()}')
# regimes for daily
fr=np.log(p.shift(-1)/p); qlist=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: qlist.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.Series(dict(qlist)); n=len(q)
for name,sub in [('early',q.iloc[:n//3]),('middle',q.iloc[n//3:2*n//3]),('late',q.iloc[2*n//3:])]: print(name, len(sub), sub.mean())
# artifacts
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_3_20301007_trend_gated_reversal5_signal.csv',index=False)
ics=q.rename('ic').reset_index(); ics.columns=['date','ic']; ics.to_csv('scripts/miner_3_20301007_trend_gated_reversal5_ic.csv',index=False)
