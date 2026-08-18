import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2032-03-04')
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date'); d=d[d.date<=cutoff][['date','close','volume']].drop_duplicates('date').set_index('date'); px[s]=d
close=pd.DataFrame({s:d.close for s,d in px.items()}).sort_index(); vol=pd.DataFrame({s:d.volume for s,d in px.items()}).reindex(close.index); ret=close.pct_change()
r5=close.pct_change(5); res=r5.sub(r5.median(axis=1),axis=0); rv=ret.rolling(40,min_periods=20).std()*np.sqrt(252); volmed=vol.rolling(20,min_periods=10).median(); shock=(vol/volmed-1).clip(-.5,2)
factor=(-res/(rv+1e-8))*(1+.35*shock.clip(lower=0))
for h in [5,10,20]:
 fr=close.shift(-h)/close.shift(-1)-1; ics=[]; cov=[]; ns=[]
 for dt in close.index:
  a=factor.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: ics.append(spearmanr(a[ok],b[ok]).statistic); cov.append(ok.mean()); ns.append(ok.sum())
 x=np.array(ics); print('horizon dates avgN IC ICIR hit coverage',(h,len(x),np.mean(ns),np.nanmean(x),np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12),np.mean(x>0),np.mean(cov)))
for days in [252,504,756]:
 fr=close.shift(-5)/close.shift(-1)-1; ics=[]
 for dt in close.index[-days:]:
  a=factor.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: ics.append(spearmanr(a[ok],b[ok]).statistic)
 x=np.array(ics); print('recent',days,len(x),np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12))
r=factor.rank(axis=1,pct=True); dates=r.index[::10]; dif=[]
for a,b in zip(dates[:-1],dates[1:]):
 z=(r.loc[b]-r.loc[a]).dropna(); dif.append(np.mean(abs(z)))
print('cutoff',close.index.max().date(),'assets',len(U),'dates',len(close),'turnover_proxy',np.nanmean(dif),'signal_coverage',factor.notna().mean().mean())
