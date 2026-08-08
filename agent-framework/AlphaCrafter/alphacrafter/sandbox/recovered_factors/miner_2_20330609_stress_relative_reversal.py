import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn); q.date=pd.to_datetime(q.date); d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2033-06-08']; r=px.pct_change()
vf='../persistent/index_data/VIX.csv'; v=pd.read_csv(vf); v.date=pd.to_datetime(v.date); v=v.set_index('date')
vc=[c for c in v.columns if c.lower() in ('close','adj_close','value','vix')]
if not vc: vc=[c for c in v.columns if c!='symbol']
vix=pd.to_numeric(v[vc[0]],errors='coerce').reindex(px.index).ffill()
# Stress-conditioned idiosyncratic reversal: fade 3-day performance relative to
# contemporaneous cross-asset mean only when VIX has risen over the prior 5 sessions.
ueq=r.mean(axis=1); resid=r.sub(ueq,axis=0)
stress=(vix.pct_change(5)>0.08).astype(float)
sig=(-resid.rolling(3,min_periods=3).sum()).mul(stress,axis=0).shift(1)
print('candidate stress_relative_reversal_3obs dates',len(px),'assets',len(px.columns),'vixcol',vc[0])
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and sig.loc[dt].abs().sum()>0:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2) if ns else 0,'IC',round(a.mean(),6) if len(a) else None,'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None,'hit',round((a>0).mean(),4) if len(a) else None)
print('coverage',round(sig.notna().mean().mean(),4),'active_date_fraction',round((sig.abs().sum(axis=1)>0).mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2033-06-08')]:
 sub=sig.loc[lo:hi]; fr=px.shift(-1)/px-1; aa=[]
 for dt in sub.index:
  z=pd.concat([sub.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and sub.loc[dt].abs().sum()>0: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 aa=np.asarray(aa); print('REG',lo,hi,'dates',len(aa),'IC',round(aa.mean(),6) if len(aa) else None,'ICIR',round(aa.mean()/aa.std(ddof=1),6) if len(aa)>1 else None)
