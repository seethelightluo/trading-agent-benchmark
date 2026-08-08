"""Enhanced scheduled revalidation: residual downside absorption quality, completed bars through 2032-10-27."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2032-10-27'); START=pd.Timestamp('2026-07-16')
D={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index() for a in AS}; idx=pd.DatetimeIndex(sorted(set.intersection(*[set(x.loc[(x.index>=START)&(x.index<=CUT)].index) for x in D.values()])))
c=pd.DataFrame({a:D[a].reindex(idx).close for a in AS}); h=pd.DataFrame({a:D[a].reindex(idx).high for a in AS}); l=pd.DataFrame({a:D[a].reindex(idx).low for a in AS})
r=c.pct_change(); m=r.median(axis=1); beta=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0); res=r-beta.mul(m,axis=0); z=res.div(res.rolling(60,min_periods=45).std()); sev=(-z).clip(lower=0,upper=4); clv=((c-l)/(h-l)).clip(0,1).replace([np.inf,-np.inf],np.nan)
def wavg(w,x,n,minn): return (w*x).rolling(n,min_periods=minn).sum()/(w.rolling(n,min_periods=minn).sum()+1e-12)
s=wavg(sev,clv,20,15)-wavg(sev,clv,60,45)
print('FACTOR residual_downside_absorption_quality_20_60obs cutoff',CUT.date(),'assets',len(AS),'calendar_dates',len(idx));print('coverage',int(s.notna().sum().sum()),'/',s.size,round(s.notna().mean().mean(),6))
O={}
for q in [1,5,10,20]:
 f=c.shift(-q)/c-1; vals=[]; ns=[]; dates=[]
 for t in idx:
  z=pd.concat([s.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(t)
 vals=np.array(vals);ns=np.array(ns);O[q]=(vals,pd.DatetimeIndex(dates),ns); print('H',q,'dates',len(vals),'IC',round(vals.mean(),6),'ICIR',round(vals.mean()/vals.std(ddof=1),6),'hit',round((vals>0).mean(),6),'mean_n',round(ns.mean(),3),'min_n',int(ns.min()),'PASS',abs(vals.mean())>=.007 and abs(vals.mean()/vals.std(ddof=1))>=.084)
x,dates,_=O[20]
for name,lo,hi in [('2026_2029','2026-07-16','2029-12-31'),('2030_to_cutoff','2030-01-01',CUT),('recent_12m','2031-10-28',CUT)]:
 q=x[(dates>=lo)&(dates<=hi)]; print('REGIME',name,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),6))
ranks=s.rank(axis=1,pct=True);print('turnover',round((ranks-ranks.shift()).abs().stack().mean(),6),'median_iqr',round((s.quantile(.75,axis=1)-s.quantile(.25,axis=1)).median(),6))
