import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for a in assets:
 q=pd.read_csv('../persistent/stock_data/'+a+'.csv'); q.date=pd.to_datetime(q.date); d[a]=q.set_index('date').close
p=pd.DataFrame(d).sort_index(); r=p.pct_change()
# Dispersion-gated idiosyncratic 3-day reversal. High cross-asset dispersion
# identifies stressed rotation days; normalization makes signals comparable across assets.
vol=r.rolling(15,min_periods=10).std()
csdisp=r.rolling(5,min_periods=4).std(axis=1)
dispbase=csdisp.rolling(60,min_periods=30).median()
gate=(csdisp/dispbase).clip(0.5,2.5)
sig=(-r.rolling(3,min_periods=3).sum()/vol*gate).shift(1)
print('range',p.index.min().date(),p.index.max().date(),'assets',len(p.columns))
print('candidate dispersion_gated_3d_reversal valid_cells',int(sig.notna().sum().sum()))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x): vals.append(x); ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4),'mean_valid',round(sig.notna().sum(axis=1).mean(),2))
for period,mask in [('2020-23',p.index<'2024-01-01'),('2024-27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('2028-30',(p.index>='2028-01-01')&(p.index<'2031-01-01')),('2031-33',p.index>='2031-01-01')]:
 vals=[]
 for dt in p.index[mask]:
  z=pd.concat([sig.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(vals); print('REGIME',period,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
