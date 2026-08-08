import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}; cs={}
for s in U:
 d=get_stock_daily_data(s,days=3000); z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); z=z.drop_duplicates('date').set_index('date').close; rr=z.pct_change(fill_method=None); br=rr.gt(0).rolling(20,min_periods=15).mean()*2-1; vo=rr.rolling(20,min_periods=15).std(); fs[s]=(z.pct_change(20,fill_method=None)*br/vo).replace([np.inf,-np.inf],np.nan); cs[s]=z
f=pd.DataFrame(fs); c=pd.DataFrame(cs)
for h in [1,5,10]:
 fw=c.pct_change(h,fill_method=None).shift(-h); by=[]; obs=0
 for dt in f.index:
  q=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna(); obs+=len(q)
  if len(q)>=8: by.append(q.x.corr(q.y))
 by=pd.Series(by).dropna(); print('horizon',h,'dates',len(by),'obs',obs,'avgN',round(obs/len(by),2),'IC %.6f ICIR %.6f hit %.4f'%(by.mean(),by.mean()/by.std(ddof=1)*np.sqrt(252),(by>0).mean()))
print('coverage',f.notna().sum().sum()/(len(f.index)*15),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',f.index.min(),f.index.max())
