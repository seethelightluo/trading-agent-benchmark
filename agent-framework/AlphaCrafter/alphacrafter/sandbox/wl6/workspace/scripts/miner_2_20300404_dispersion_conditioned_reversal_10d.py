import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; end=pd.Timestamp('2030-04-04')
raw=[]
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date'])[['date','close']].rename(columns={'close':s}).set_index('date'); raw.append(d)
p=raw[0]
for d in raw[1:]: p=p.join(d,how='inner')
p=p[p.index<=end].sort_index(); rets=p.pct_change(); r5=p/p.shift(5)-1; vol20=rets.rolling(20).std()*np.sqrt(252); disp=rets.rolling(5).std().mean(axis=1)
ics=[]; years={}; turns=[]
for i in range(126,len(p)-10):
 dt=p.index[i]; med=disp.iloc[i-126:i].median()
 if not np.isfinite(med) or med<=0: continue
 f=(-r5.iloc[i]/vol20.iloc[i])*np.clip(disp.iloc[i]/med,.5,2); fr=p.iloc[i+10]/p.iloc[i]-1; ok=f.notna()&fr.notna();
 if ok.sum()<8: continue
 q=spearmanr(f[ok],fr[ok]).statistic
 if np.isfinite(q): ics.append(q); years.setdefault(dt.year,[]).append(q); turns.append(np.mean(np.abs(f[ok].rank(pct=True)-.5)))
print('dates',len(ics),'avg_n',p.shape[1],'coverage',1.0); a=np.array(ics); print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover_proxy',np.mean(turns))
for h in [5,10,20]:
 z=[]
 for i in range(126,len(p)-h):
  me=disp.iloc[i-126:i].median()
  if not np.isfinite(me) or me<=0: continue
  f=(-r5.iloc[i]/vol20.iloc[i])*np.clip(disp.iloc[i]/me,.5,2); fr=p.iloc[i+h]/p.iloc[i]-1; ok=f.notna()&fr.notna()
  if ok.sum()>=8: z.append(spearmanr(f[ok],fr[ok]).statistic)
 print('decay',h,np.nanmean(z),len(z))
print('regimes',[(y,len(v),round(np.mean(v),5)) for y,v in sorted(years.items())])
