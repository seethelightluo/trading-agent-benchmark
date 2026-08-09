import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; macro='../persistent/index_data'; cut=pd.Timestamp('2026-11-19')
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float); px[s]=d[d.index<=cut]
P=pd.DataFrame(px).sort_index(); P.index=pd.to_datetime(P.index); R=P.pct_change()
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float); v=v[v.index<=cut]; v.index=pd.to_datetime(v.index)
v5=v.pct_change(5).reindex(P.index).ffill()
r5=P.pct_change(5); shock=v5.clip(-0.5,0.5)
f=(-r5.mul(1+shock,axis=0)).where(shock>0, -r5*0.5)
for h in [1,5,10]:
 out=[]; Y=P.shift(-h).div(P)-1
 for dt in P.index:
  q=pd.concat([f.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8: out.append((dt,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('year',yr,'IC',g.mean(),'n',len(g))
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'rankturn',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('highVIX dates',(shock>0).sum(),'of',shock.notna().sum())
