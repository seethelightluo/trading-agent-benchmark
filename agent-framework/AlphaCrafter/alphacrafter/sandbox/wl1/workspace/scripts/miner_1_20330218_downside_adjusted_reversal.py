import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); bench=R.mean(axis=1)
res=R.sub(bench,axis=0)
# Downside-risk adjusted relative reversal: fade 30d residual losses, scaled by 40d downside deviation; lagged.
down=res.clip(upper=0).rolling(40,min_periods=25).std()
f=-(res.rolling(30,min_periods=25).sum()/down.replace(0,np.nan)).shift(1)
fr=P.shift(-10)/P-1
ics=[]; rows=[]
for dt in f.index:
 ok=f.loc[dt].notna()&fr.loc[dt].notna()
 if ok.sum()>=8:
  c=f.loc[dt,ok].corr(fr.loc[dt,ok])
  if np.isfinite(c): ics.append(c); rows.append((dt,c,int(ok.sum())))
a=np.array(ics); ranks=f.rank(axis=1,pct=True); tr=[]
for i in range(1,len(ranks)):
 ok=ranks.iloc[i].notna()&ranks.iloc[i-1].notna()
 if ok.sum()>=8: tr.append((ranks.iloc[i][ok]-ranks.iloc[i-1][ok]).abs().mean())
print('candidate downside_adjusted_residual_reversal_30d')
print('dates',len(a),'avg_instruments',np.mean([n for d,c,n in rows]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.mean(tr),'coverage',f.notna().mean().mean())
for name,lo,hi in [('2024-26','2024','2026-12-31'),('2027-29','2027','2029-12-31'),('2030-32','2030','2032-12-31'),('2033','2033','2033-01-31')]:
 q=np.array([c for d,c,n in rows if str(d)>=lo and str(d)<=hi]); print(name,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
out=pd.DataFrame([(d.strftime('%Y-%m-%d'),s,float(f.loc[d,s]) if pd.notna(f.loc[d,s]) else np.nan) for d in f.index for s in U],columns=['date','symbol','signal']); out.to_csv('scripts/miner_1_20330218_downside_adjusted_reversal_signal.csv',index=False)
