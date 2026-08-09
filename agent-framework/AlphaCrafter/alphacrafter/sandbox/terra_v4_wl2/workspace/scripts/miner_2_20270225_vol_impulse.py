import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for s in U}
P=pd.DataFrame({s:d.close for s,d in D.items()}); R=P.pct_change()
# Volatility impulse: negative of recent volatility expansion, cross-sectionally ranked.
# Assets whose 5d realized volatility jumped most relative to their 30d baseline are expected to mean-revert.
rv5=R.rolling(5,min_periods=5).std(); rv30=R.rolling(30,min_periods=30).std()
f=-(rv5/rv30.replace(0,np.nan)-1.0)
y={h:P.shift(-h)/P-1 for h in [1,5,10]}
rows=[]; sig=[]
for dt in P.index:
 v=f.loc[dt]
 for h in [1,5,10]:
  q=pd.DataFrame({'f':v,'y':y[h].loc[dt]}).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: rows.append((dt,h,spearmanr(q.f,q.y).statistic,len(q)))
 sig.extend([(dt,s,f.loc[dt,s]) for s in U])
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=df[df.h==h]; z=q.ic.to_numpy(); print('H',h,'dates',len(q),'avg_names',round(q.n.mean(),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  x=q.set_index('date').loc[lo:hi].ic; print(' regime',lo,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']); out.to_csv('../persistent/factor_signals_miner_2_20270225_vol_impulse.csv',index=False)
w=out.pivot(index='date',columns='asset',values='signal'); print('coverage',round(out.signal.notna().mean(),4),'artifact_rows',len(out),'turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
