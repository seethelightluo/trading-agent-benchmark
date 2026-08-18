import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-05-12'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.loc[:cut,'close']
# common date panel, factor lagged/observable at date t; 10d forward
P=pd.DataFrame(px).dropna(how='all'); R=P.pct_change()
# Trend-quality: 40d compounded return, multiplied by directional consistency, divided by realized vol; all trailing
ret=P.pct_change(40); vol=R.rolling(40).std()*np.sqrt(252); consistency=R.gt(0).rolling(40).mean()
f=ret/vol*consistency
ics=[]; turns=[]; ns=[]
for i in range(len(P)-10):
 dt=P.index[i]
 vals=f.iloc[i]; fw=P.iloc[i+10]/P.iloc[i]-1
 z=pd.concat([vals.rename('f'),fw.rename('r')],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.f,z.r).statistic); ns.append(len(z))
# turnover rank changes daily
ranks=f.rank(axis=1,pct=True); turns=ranks.diff().abs().mean(axis=1).dropna()
a=np.array(ics); print('cutoff',cut.date(),'dates',len(a),'avgN',np.mean(ns),'minN',min(ns),'coverage',np.mean([n/15 for n in ns])); print('H10 IC %.8f ICIR %.8f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a)),np.mean(a>0))); print('turnover %.6f'%turns.mean())
for n in [260,520,780]:
 x=a[-n:]; print('recent',n,'IC %.8f ICIR %.8f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1)*np.sqrt(len(x))))
# other horizons
for h in [5,20]:
 aa=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('r')],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.f,z.r).statistic)
 x=np.array(aa); print('H%d dates %d IC %.8f ICIR %.8f'%(h,len(x),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(len(x))))
