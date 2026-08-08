import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];d={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  q=pd.read_csv(p);q.date=pd.to_datetime(q.date);d[a]=q.set_index('date').close
px=pd.DataFrame(d).sort_index();r=px.pct_change()
def rv(n): return r.rolling(n,min_periods=max(4,n-1)).std()
def mom(n): return r.rolling(n,min_periods=max(4,n-1)).sum()
def pos(n):
 hi=px.rolling(n,min_periods=max(20,n//2)).max();lo=px.rolling(n,min_periods=max(20,n//2)).min();return (px-lo)/(hi-lo).replace(0,np.nan)-.5
# dispersion-conditioned reversal, cross-sectional residualized against reversal and momentum proxies
cs5=mom(5); disp=cs5.std(axis=1).rolling(20,min_periods=15).rank(pct=True)
raw=(-r/rv(20)).mul(disp,axis=0).shift(1)
controls=pd.concat([(-r).stack(),(-mom(5)/rv(5)).stack(),(mom(20)/rv(20)).stack()],axis=1)
controls.columns=['rawrev','volrev','trend']; out=pd.DataFrame(index=raw.index,columns=raw.columns,dtype=float)
for dt in raw.index:
 y=raw.loc[dt]; X=controls.loc[dt] if dt in controls.index else None
 # controls construction below is replaced by same-date cross-section
 vals=pd.DataFrame({'y':y,'rawrev':(-r.loc[dt]),'volrev':(-mom(5).loc[dt]/rv(5).loc[dt]),'trend':(mom(20).loc[dt]/rv(20).loc[dt])}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(vals)>=8:
  xx=np.column_stack([np.ones(len(vals)),vals[['rawrev','volrev','trend']].values]); beta=np.linalg.lstsq(xx,vals.y.values,rcond=None)[0]; out.loc[dt,vals.index]=vals.y.values-xx@beta
cand=out
fr=px.pct_change().shift(-1);v=[];ns=[];dates=[]
for dt in px.index:
 z=pd.concat([cand.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):v.append(q);ns.append(len(z));dates.append(dt)
s=pd.Series(v,index=dates)
print('assets',len(d),'rows',len(px),'coverage',cand.notna().sum().sum()/cand.size)
print('daily dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'turnover',round(cand.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for h in [5,10,20]:
 fr=px.pct_change(h).shift(-h);z=[]
 for dt in px.index:
  q=pd.concat([cand.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=pd.Series(z).dropna();print('h',h,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('regimes')
for y,g in s.groupby(s.index.year):print(y,len(g),round(g.mean(),6),round(g.mean()/g.std(ddof=1),6))
libs={'ravmom20':mom(20)/rv(20),'volnormrev5':-mom(5)/rv(5),'risktrend20':mom(20)/rv(20),'rangeacc':pos(20)-pos(60),'voltransition':rv(5)/rv(60),'volshock':mom(5)*(rv(5)/rv(40)-1),'rawmom20':mom(20),'rawrev1':-r}
cor=[]
for k,x in libs.items():
 z=pd.concat([cand.stack(),x.reindex_like(cand).stack()],axis=1).dropna();rho=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic;print('rho',k,round(rho,6),'n',len(z));cor.append(abs(rho))
print('max_abs_library_correlation',round(max(cor),6))
