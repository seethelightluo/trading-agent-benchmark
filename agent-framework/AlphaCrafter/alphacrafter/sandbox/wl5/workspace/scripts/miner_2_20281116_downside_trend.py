import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: medium-term trend weighted by breadth/persistence, normalized by downside risk.
px={}
for s in U:
 d=get_stock_daily_data(s, days=2600)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill()
ret=p.pct_change()
# 40d return, persistence of positive daily returns, downside volatility
mom=p/p.shift(30)-1
persist=ret.rolling(60).mean()/(ret.rolling(60).std()+1e-12)
down=np.sqrt((ret.clip(upper=0).fillna(0)**2).rolling(40, min_periods=20).mean())*np.sqrt(252)
raw=mom*(1+0.35*np.tanh(persist))/down.replace(0,np.nan)
# cross sectional standardized ranks (higher is better)
f=raw.sub(raw.mean(axis=1),axis=0)
rows=[]
for h in [5,10,20]:
  fr=f
  # avoid lookahead: factor at t predicts t+h close return from t to t+h
  fw=p.shift(-h)/p-1
  ics=[]
  for dt in f.index:
   x=fr.loc[dt]; y=fw.loc[dt]
   z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  a=pd.Series(ics).dropna()
  rows.append((h,len(a),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))
# rank turnover daily and coverage
r=f.rank(axis=1,pct=True)
to=(r-r.shift(1)).abs().mean(axis=1).mean()
coverage=f.notna().sum(axis=1).mean()/len(U)
print('rows',len(p),'dates',p.index.min(),p.index.max(),'assets',len(px),'coverage',coverage,'turnover',to)
for x in rows: print('H',x[0],'N',x[1],'IC %.6f ICIR %.6f hit %.4f'%x[2:])
# regime split 10d
h=10; fw=p.shift(-h)/p-1
ics=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(ics,columns=['date','ic']).set_index('date').dropna()
for name, q in [('2020-24',a.loc[:'2024-12-31']),('2025-26',a.loc['2025-01-01':'2026-12-31']),('2027-28',a.loc['2027-01-01':])]:
 print(name,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12)))
# artifact
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_2_20281116_downside_trend_signal.csv')
