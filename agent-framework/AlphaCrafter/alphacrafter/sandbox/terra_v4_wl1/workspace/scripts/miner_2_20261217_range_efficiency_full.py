import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').sort_values('date'); r=d.close.pct_change(); f=d.close.pct_change(20)/(r.abs().rolling(20,min_periods=15).sum()+1e-12).shift(1)
 for dt,a,*b in zip(d.date,f,d.close.shift(-1)/d.close-1,d.close.shift(-5)/d.close-1,d.close.shift(-10)/d.close-1): rows.append((dt,s,a,*b))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd1','fwd5','fwd10']); z=x.dropna(subset=['factor']);
for h in ['fwd1','fwd5','fwd10']:
 q=[];ns=[]
 for dt,g in z.dropna(subset=[h]).groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g[h].nunique()>1:q.append(spearmanr(g.factor,g[h]).statistic);ns.append(len(g))
 q=pd.Series(q);print(h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
 for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
  v=[]
  for dt,g in z[(z.date.dt.year>=a)&(z.date.dt.year<=b)].dropna(subset=[h]).groupby('date'):
   if len(g)>=8 and g.factor.nunique()>1 and g[h].nunique()>1:v.append(spearmanr(g.factor,g[h]).statistic)
  v=pd.Series(v);print(' ',a,b,len(v),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6))
# correlation against residual artifact, aligned signal values
f=x.pivot(index='date',columns='symbol',values='factor')
r=pd.read_csv('scripts/miner_2_20261217_residual_reversal_signal.csv',index_col=0,parse_dates=True)
if set(f.columns)&set(r.columns):
 a,b=f.align(r,join='inner',axis=0); vals=[]
 for dt in a.index:
  aa=a.loc[dt];bb=b.loc[dt]; ok=aa.notna()&bb.notna()
  if ok.sum()>=8: vals.append(aa[ok].corr(bb[ok],method='spearman'))
 print('library_corr_mean_abs',np.nanmean(np.abs(vals)),'max_abs',np.nanmax(np.abs(vals)),'dates',len(vals))
f.to_csv('scripts/miner_2_20261217_range_efficiency_signal.csv')
print('coverage',z.shape[0]/x.shape[0],'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
