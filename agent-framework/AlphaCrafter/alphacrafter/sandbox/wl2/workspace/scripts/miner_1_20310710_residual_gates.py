import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=4100)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=4100)
 if x is not None:D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); y=r.shift(-1); m=r.mean(1); res=r.sub(m,axis=0); rv=res.rolling(20,min_periods=10).std(); raw=-(res.rolling(3,min_periods=3).sum()/rv)
disp=r.std(1); med=disp.rolling(252,min_periods=100).median(); vix=None
try:
 vix=get_index_daily_data('VIX',days=4100).set_index('date').close.astype(float).reindex(p.index).ffill()
except: pass
gates={'uncond':pd.Series(True,index=p.index),'high_disp':disp>med,'contract':(disp>med)&(disp<disp.shift(5)),'low_disp':disp<med,'market_down':m<0,'vix_high':(vix>vix.rolling(252,min_periods=100).quantile(.7)) if vix is not None else pd.Series(False,index=p.index)}
for name,g in gates.items():
 f=raw.where(g,np.nan); rows=[]
 for i in range(len(f)-1):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:rows.append((p.index[i],z.f.corr(z.y),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');
 if len(q)>1:print(name,'dates',len(q),'active',round(g.mean(),3),'N',round(q.n.mean(),2),'IC %.6f IR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
