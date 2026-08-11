import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Medium-horizon trend persistence: 10d risk-adjusted momentum, attenuated when 10d and 60d directions disagree.
vol=r.rolling(20,min_periods=15).std(); m10=p.pct_change(10); m60=p.pct_change(60)
agree=np.where(np.sign(m10)==np.sign(m60),1.0,0.35)
f=(m10/vol.replace(0,np.nan)*agree).shift(1)
def ev(h,sl=slice(None)):
 y=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.loc[sl].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 a=np.array(vals); return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
for h in [1,3,5,10]: print('h',h,ev(h))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5),'dates',len(p),'instruments',len(D))
for n,s in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]: print(n,ev(10,s))
# decay from current recent blocks
for n,s in [('2027',slice('2027','2027')),('2028',slice('2028',None))]: print('recent',n,ev(10,s))
