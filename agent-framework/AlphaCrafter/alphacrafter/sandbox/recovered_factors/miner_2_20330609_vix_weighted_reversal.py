import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
K=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];d={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]
 if s in K:
  q=pd.read_csv(f);q.date=pd.to_datetime(q.date);d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2033-06-08'];r=px.pct_change();q=pd.read_csv('../persistent/index_data/VIX.csv');q.date=pd.to_datetime(q.date);v=q.set_index('date');c=[x for x in v if x.lower()=='close'][0];vix=pd.to_numeric(v[c],errors='coerce').reindex(px.index).ffill()
res=r.sub(r.mean(axis=1),axis=0); base=-res.rolling(3,min_periods=3).sum(); mag=vix.pct_change(5).clip(lower=0); sig=base.mul(mag,axis=0).shift(1)
print('candidate vix_magnitude_weighted_relative_reversal',len(px),len(px.columns))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1;a=[];n=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and sig.loc[dt].abs().sum()>0:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'N',round(np.mean(n),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(sig.notna().mean().mean(),4),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
