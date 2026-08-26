import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d)>100:
        x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x.set_index('date').close
P=pd.DataFrame(px).sort_index()
# lagged trend blend: signal on t uses prices through t-1, 20/60d returns, volatility scale
r1=P.shift(1)/P.shift(21)-1
r2=P.shift(1)/P.shift(61)-1
vol=P.pct_change().rolling(40).std().shift(1)*np.sqrt(40)
sig=(0.65*r1+0.35*r2)/vol.replace(0,np.nan)
# cross-sectional rank/median center, preserving lag
sig=sig.sub(sig.median(axis=1),axis=0)
fwd=P.shift(-1)/P-1
rows=[]; daily=[]
for dt in sig.index:
    z=sig.loc[dt]; y=fwd.loc[dt]; v=z.notna()&y.notna()
    if v.sum()>=8:
        ic=z[v].corr(y[v],method='spearman'); rows.append(ic)
        daily.append((dt,ic,int(v.sum())))
a=pd.Series(rows)
print('rows',len(P),'assets',len(P.columns),'dates',len(a),'avg_n',np.mean([x[2] for x in daily]))
print('IC %.8f ICIR %.8f hit %.5f' % (a.mean(),a.mean()/a.std(ddof=1), (a>0).mean()))
for h in [5,10,20]:
    y=P.shift(-h)/P-1; vals=[]; ns=[]
    for dt in sig.index:
      v=sig.loc[dt].notna()&y.loc[dt].notna()
      if v.sum()>=8: vals.append(sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'));ns.append(v.sum())
    q=pd.Series(vals); print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',np.mean(ns))
out=pd.DataFrame(daily,columns=['date','ic','n']); out.to_csv('scripts/miner_3_20310210_trendblend_ic.csv',index=False)
ss=sig.stack().rename('signal').reset_index(); ss.columns=['date','symbol','signal']; ss.to_csv('scripts/miner_3_20310210_trendblend_signal.csv',index=False)
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
print('regimes', [a.iloc[i:j].mean() for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
