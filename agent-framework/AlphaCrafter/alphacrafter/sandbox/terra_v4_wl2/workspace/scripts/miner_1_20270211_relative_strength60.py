import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if os.path.exists(p):
  x=pd.read_csv(p,parse_dates=['date']).set_index('date'); D[s]=x['close']
px=pd.DataFrame(D).sort_index(); r=px.pct_change()
# Relative strength: 60d return demeaned by contemporaneous cross-section, intended to remove common beta
f=px.pct_change(60); f=f.sub(f.mean(axis=1),axis=0)
fr=r.shift(-1)
ics=[]; turnovers=[]; ns=[]
prev=None
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  rank=z.iloc[:,0].rank(pct=True)
  if prev is not None: turnovers.append(np.mean(abs(rank-prev)))
  prev=rank
ics=np.array(ics)
print('dates',len(ics),'avg_n',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f turnover %.5f coverage %.4f'%(np.nanmean(ics),np.nanmean(ics)/np.nanstd(ics,ddof=1),np.mean(ics>0),np.nanmean(turnovers),f.notna().sum().sum()/f.size))
for h in [5,10,20]:
 ff=px.pct_change(h); ff=ff.sub(ff.mean(axis=1),axis=0); target=px.pct_change(h).shift(-h)
 q=[]
 for dt in ff.index:
  z=pd.concat([ff.loc[dt],target.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(h,'d',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1))
# regime splits
for name,mask in [('2020-22',f.index<'2023-01-01'),('2023-24',(f.index>='2023-01-01')&(f.index<'2025-01-01')),('2025+',f.index>='2025-01-01')]:
 q=[]
 for dt in f.index[mask]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(name,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
# artifact
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('../persistent/factor_signals_miner_1_20270211_relative_strength60.csv')
