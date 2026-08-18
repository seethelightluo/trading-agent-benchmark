import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 D[s]=d['close'].replace(0,np.nan)
px=pd.DataFrame(D).sort_index(); px=px.loc[:'2033-07-22']
# Trend-agreement momentum: medium momentum amplified only when 20d and 60d trends agree
r20=px.pct_change(20); r60=px.pct_change(60)
sig=(r20 * np.sign(r20*r60)).shift(1) # positive when trend agrees, negative when disagreement
fwd=px.pct_change(1).shift(-1)
ics=[]; dates=[]; nobs=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): ics.append(ic); dates.append(dt); nobs.append(len(z))
ics=np.array(ics)
print('idea=agreement-gated 20d momentum; dates',len(ics),'range',dates[0],dates[-1],'avg_n',np.mean(nobs),'coverage',np.mean(nobs)/15)
print('IC %.8f ICIR %.8f hit %.4f'%(np.mean(ics),np.mean(ics)/np.std(ics,ddof=1),np.mean(ics>0)))
for a,b in [('2020-01-01','2025-12-31'),('2026-01-01','2029-12-31'),('2030-01-01','2033-07-22')]:
 q=ics[[a<=str(d.date())<=b for d in dates]]
 print(a,b,'n',len(q),'IC',np.mean(q) if len(q) else np.nan,'IR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 yy=px.pct_change(h).shift(-h); q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('h',h,'n',len(q),'IC',np.nanmean(q),'IR',np.nanmean(q)/np.nanstd(q,ddof=1))
# turnover proxy rank top/bottom direction changes
rank=sig.rank(axis=1,pct=True); print('turnover_proxy',rank.diff().abs().mean().mean())
# artifact for reproducibility
out=pd.DataFrame(sig); out.index.name='date'; out.to_csv('factors/miner_3_20330722_agreement20_signal.csv')
