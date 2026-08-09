import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
F={}; Y={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 p=d.close; prev=p.shift(1); tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1); atr=tr.rolling(20,min_periods=15).mean()
 shock=((p-d.open)/atr).replace([np.inf,-np.inf],np.nan)
 F[s]=-shock.shift(1).ewm(span=3,adjust=False,min_periods=3).mean()
 for h in [1,5,10]: Y.setdefault(h,{})[s]=p.shift(-h).div(p)-1
f=pd.DataFrame(F); print('dates',len(f),'assets',len(U),'coverage',f.notna().sum().sum()/f.size)
for h in [1,5,10]:
 y=pd.DataFrame(Y[h]); rows=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,len(g),round(g.mean(),5),round(g.mean()/g.std(ddof=1),4))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean()); print('period',f.index.min(),f.index.max())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20261217_smoothed_atr_intraday_signal.csv',index=False)
