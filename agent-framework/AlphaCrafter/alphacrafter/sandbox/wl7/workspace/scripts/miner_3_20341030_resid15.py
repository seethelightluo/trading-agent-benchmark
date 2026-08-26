import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2034-10-29')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close
px=pd.concat([load(s).rename(s) for s in U],axis=1).sort_index().loc[:END]; r=px.pct_change(); med=r.median(axis=1); resid=r.sub(med,axis=0)
# Medium-horizon residual mean reversion, volatility normalized; lagged signal is used at decision time
f=-resid.rolling(15,min_periods=15).sum()/resid.rolling(60,min_periods=40).std()
print('DATES',px.index.min().date(),px.index.max().date(),'ASSETS',px.shape[1])
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h); z=[];ds=[];ns=[]
 for dt in px.index:
  a=f.loc[dt];b=y.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic);ds.append(dt);ns.append(ok.sum())
 z=pd.Series(z,index=ds).dropna();print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),len(z),np.mean(ns[:len(z)]),(z>0).mean()))
 if h==10:
  for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
   q=z.loc[a:b]; print('REG',a,b,'IC %.6f ICIR %.4f N %d hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),len(q),(q>0).mean()))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal');out.to_csv('scripts/miner_3_20341030_resid15_signal.csv',index=False);print('COVER',f.notna().mean().mean(),'TURN',f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
