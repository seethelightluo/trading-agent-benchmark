import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in assets}
# Breadth-conditioned risk-adjusted momentum: use only completed-day information.
# In broad positive breadth, favor persistent momentum; in negative breadth, favor relative rebound (reversal).
cl=pd.concat({a:d.close for a,d in data.items()},axis=1).sort_index()
r=cl.pct_change(); mom=r.rolling(10,min_periods=8).sum(); vol=r.rolling(20,min_periods=12).std()
breadth=(mom>0).mean(axis=1)
# continuous regime gate, centered at neutral breadth; factor is risk-adjusted momentum when breadth positive,
# and risk-adjusted short-term reversal when breadth negative.
base=mom/vol.replace(0,np.nan)
short=r.rolling(3,min_periods=2).sum()/vol.replace(0,np.nan)
gate=(breadth-0.5)*2
fac=base*gate + (-short)*(1-gate.abs())
rows=[]
for a in assets:
 z=pd.DataFrame({'f':fac[a], 'r5':cl[a].shift(-5)/cl[a]-1,'r1':cl[a].shift(-1)/cl[a]-1,'r10':cl[a].shift(-10)/cl[a]-1,'r20':cl[a].shift(-20)/cl[a]-1}).dropna(); rows.append(z.assign(asset=a))
x=pd.concat(rows); dates=x.index.unique(); print('candidate=breadth_conditioned_regime_momentum; dates',len(dates),'cells',len(x),'coverage',len(x)/(len(dates)*15))
for col in ['r1','r5','r10','r20']:
 ii=[]; ns=[]
 for d,g in x.groupby(level=0):
  g=g[['f',col]].dropna()
  if len(g)>=8: ii.append(spearmanr(g.f,g[col]).statistic); ns.append(len(g))
 ii=np.array(ii); print(col,'dates',len(ii),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(ii),np.mean(ii)/np.std(ii,ddof=1),np.mean(ii>0)))
# regimes at H5
for name,mask in [('2020-23',x.index<'2024-01-01'),('2024-27',(x.index>='2024-01-01')&(x.index<'2028-01-01')),('2028+',x.index>='2028-01-01'),('latest120',x.index>=dates.sort_values()[-120])]:
 ii=[]
 for d,g in x[mask].groupby(level=0):
  if len(g)>=8: ii.append(spearmanr(g.f,g.r5).statistic)
 ii=np.array(ii); print(name,'dates',len(ii),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(ii),np.mean(ii)/np.std(ii,ddof=1),np.mean(ii>0)))
wide=x.reset_index().pivot(index='date',columns='asset',values='f'); ranks=wide.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean().mean())
# decay by 1..20 already above
