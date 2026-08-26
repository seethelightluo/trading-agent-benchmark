import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-12-10')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close
px=pd.concat([load(s).rename(s) for s in U],axis=1).sort_index().loc[:END]
r=px.pct_change(); res=r.sub(r.median(axis=1),axis=0)
f=-res.rolling(20,min_periods=20).sum()/res.rolling(60,min_periods=40).std()
print('DATES',px.index.min().date(),px.index.max().date(),'ASSETS',px.shape[1])
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h); z=[];ds=[];ns=[]
 for dt in px.index:
  a=f.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   z.append(float(a[ok].corr(b[ok],method='spearman'))); ds.append(dt); ns.append(ok.sum())
 z=pd.Series(z,index=ds).dropna(); print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),len(z),np.mean(ns),(z>0).mean()))
 if h==10:
  for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034'),('2034-06','2034-12')]:
   q=z.loc[a:b]
   if len(q): print('REG',a,b,'IC %.6f ICIR %.4f N %d hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),len(q),(q>0).mean()))
f.to_csv('scripts/miner_1_20341211_resid20_signal.csv'); print('COVER',f.notna().mean().mean(),'TURN',f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
