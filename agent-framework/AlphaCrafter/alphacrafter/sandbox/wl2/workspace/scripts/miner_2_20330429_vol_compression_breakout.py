import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-04-29')
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Candidate: lagged volatility-compression breakout: 10d momentum scaled by 20d vol,
# active only where recent 5d volatility is below 20d volatility (pre-breakout continuation).
vol20=r.rolling(20).std(); vol5=r.rolling(5).std(); mom10=p.pct_change(10)
f=(mom10/vol20).where(vol5 < vol20)
ics=[]; rows=[]
for dt in f.index:
 if dt not in r.index: continue
 vals=f.loc[dt]; fr=r.shift(-1).loc[dt]
 z=pd.concat([vals,fr],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  ics.append(ic); rows.append((dt,ic,len(z)))
ics=np.array(ics); print('dates',len(ics),'assets',len(assets),'avg_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0))
for h in [3,5,10]:
 fr=p.pct_change(h).shift(-h)
 a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1), 'n',len(a))
for start,end in [('2020-01-01','2025-12-31'),('2026-01-01','2029-12-31'),('2030-01-01','2033-04-29')]:
 a=[ic for dt,ic,n in rows if pd.Timestamp(start)<=dt<=pd.Timestamp(end)]
 print(start,end,len(a),np.mean(a) if a else np.nan,np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan)
# turnover proxy rank overlap daily among active signals
ranks=f.rank(axis=1,pct=True); turnover=(ranks.diff().abs().mean(axis=1)).mean();print('turnover',turnover)
print('last',f.tail(1).T.dropna().to_dict())
