import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.drop_duplicates('date').set_index('date').close
close=pd.DataFrame(px).sort_index(); r=close.pct_change()
# downside-adjusted reversal: contrarian trailing return scaled by downside risk
dn=r.where(r<0).abs().rolling(20,min_periods=3).mean()
f=(-close.pct_change(10)/dn).replace([np.inf,-np.inf],np.nan)
# library proxies for redundancy, measured on pooled date/asset ranks
proxies={'reversal5':-close.pct_change(5),'leadlag5':close.pct_change(5).sub(close.pct_change(5).median(axis=1),axis=0),'momentum20':close.pct_change(20)/r.rolling(20).std()}
for h in [1,5,10,20]:
 fw=close.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r));ns.append(len(a));dates.append(dt)
 ic=pd.Series(vals,index=dates).dropna(); print('h',h,'dates',len(ic),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1)*np.sqrt(252),(ic>0).mean()))
if True:
 rr=f.rank(axis=1,pct=True); print('turnover',rr.diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum().sum()/f.size)
 for n,p in proxies.items():
  a=pd.concat([f.stack(),p.stack()],axis=1).dropna(); print('corr',n,a.iloc[:,0].corr(a.iloc[:,1]))
 for label,mask in [('early',f.index<'2023-01-01'),('mid',(f.index>='2023-01-01')&(f.index<'2025-01-01')),('late',f.index>='2025-01-01')]:
  z=[]
  fw=close.pct_change().shift(-1)
  for dt in f.index[mask]:
   a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
   if len(a)>=8:z.append(a.f.corr(a.r))
  z=pd.Series(z).dropna();print(label,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan)
print('period',close.index.min(),close.index.max(),'instruments',len(U))
