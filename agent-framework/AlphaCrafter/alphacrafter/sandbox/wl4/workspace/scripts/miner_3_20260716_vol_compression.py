import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); z=z.drop_duplicates('date').set_index('date').close
 px[s]=z
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# volatility compression: negative short/long vol, cross sectionally predictive low-risk anomaly
f=(ret.rolling(5).std()/ret.rolling(60).std()).replace([np.inf,-np.inf],np.nan)*-1
# forward returns, explicitly align and stack
for h in [1,5,10]:
 fw=close.pct_change(h).shift(-h)
 vals=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r))
 ic=pd.Series(vals).dropna(); print('h',h,'dates',len(ic),'meanN',f.notna().sum(axis=1).mean(),'IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1)*np.sqrt(252), (ic>0).mean()))
# daily rank turnover
r=f.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum().sum()/(f.size),'period',close.index.min(),close.index.max())
