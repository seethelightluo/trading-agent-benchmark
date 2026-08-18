import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000); z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.drop_duplicates('date').set_index('date').close
macro=pd.read_csv('../persistent/index_data/DXY.csv')
macro['date']=pd.to_datetime(macro['date']); dxy=macro.drop_duplicates('date').set_index('date')['close']
close=pd.DataFrame(px).sort_index().reindex(dxy.index); ret=close.pct_change(); mr=dxy.pct_change()
# DXY sensitivity: assets with negative/low beta to dollar shocks are defensive; use negative rolling beta as score
m1=ret.rolling(60,min_periods=45).mean(); mm=mr.rolling(60,min_periods=45).mean(); cov=(ret.mul(mr,axis=0).rolling(60,min_periods=45).mean()-m1.mul(mm,axis=0)); var=mr.rolling(60,min_periods=45).var(); f=(-cov.div(var,axis=0)).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 fw=close.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r)); ns.append(len(a))
 ic=pd.Series(vals).dropna(); print('h',h,'dates',len(ic),'meanN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1)*np.sqrt(252), (ic>0).mean()))
# annual regime daily
fw=close.pct_change().shift(-1); vals=[]
for dt in f.index:
 a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
 if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append((dt,a.f.corr(a.r)))
ic=pd.Series([v for _,v in vals],index=pd.to_datetime([d for d,_ in vals])); print('regime',ic.groupby(ic.index.year).agg(['mean','count']).round(5).to_dict())
r=f.rank(axis=1,pct=True); print('turnover %.6f'%r.diff().abs().mean(axis=1).mean(),'coverage %.4f'%(f.notna().sum().sum()/f.size),'period',close.index.min(),close.index.max())
# library rank correlation audit where possible
for name in ['peer_median_leadlag_5d','short_term_reversal_5d','risk_adjusted_momentum_20d']:
 print('library',name,'not computed')
