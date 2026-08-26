import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[s]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); up=r.clip(lower=0).rolling(40,min_periods=30).mean(); dn=(-r.clip(upper=0)).rolling(40,min_periods=30).mean(); vol=r.rolling(40,min_periods=30).std()
f=-(up/(dn+1e-5)-1)/(vol*np.sqrt(252)+.10)
rows={h:[] for h in [10,20,40,60]}; ns=[]; dates=[]
for i,t in enumerate(P.index):
 if t<pd.Timestamp('2024-01-01'): continue
 for h in rows:
  if i+h>=len(P): continue
  z=pd.concat([f.loc[t],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   rows[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
   if h==60: ns.append(len(z)); dates.append(t)
print('factor=40D downside/upside asymmetry, volatility scaled (sign chosen contrarian)'); print('dates',len(dates),'avgN',np.mean(ns),'coverage',f.notna().mean().mean(),'assets',len(U))
for h,x in rows.items():
 a=np.array(x); print(h,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)+1e-12),'hit',np.mean(a>0))
for label,a,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031-32','2031-01-01','2032-12-31'),('2033','2033-01-01','2033-12-31')]:
 q=[v for t,v in zip(dates,rows[60]) if pd.Timestamp(a)<=t<=pd.Timestamp(b)]; print('regime',label,'n',len(q),'IC',np.mean(q) if q else np.nan)
os.makedirs('scripts',exist_ok=True); out=f.loc[dates].copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20330623_downside_asymmetry_signal.csv')
