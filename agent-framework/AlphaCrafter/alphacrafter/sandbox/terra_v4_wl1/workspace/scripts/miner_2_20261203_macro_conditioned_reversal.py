import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-02')
base='../persistent/stock_data'; macro='../persistent/index_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float) for s in U}
P=pd.DataFrame(px).sort_index(); P=P[P.index<=cut]; v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float);v=v[v.index<=cut]
r5=P.pct_change(5); shock=v.pct_change(5).reindex(P.index).ffill(); f=(-r5.mul(1+shock.clip(-.5,.5),axis=0)).where(shock>0,-r5*.5); basef=-r5
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for dt in P.index:
  q=pd.concat([f.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((dt,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1:
  print('years',[(int(y),round(g.mean(),5),len(g)) for y,g in ic.groupby(ic.index.year)])
 print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'corr_reversal',round(f.stack().corr(basef.stack()),4))
print('cut',P.index.max())
