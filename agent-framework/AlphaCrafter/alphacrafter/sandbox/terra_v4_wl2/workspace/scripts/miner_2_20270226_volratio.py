import pandas as pd, numpy as np, glob, json, os
from scipy.stats import spearmanr
cut='2027-02-24'
files=glob.glob('../persistent/stock_data/*.csv')
prices={}
for p in files:
 d=pd.read_csv(p,usecols=['date','close']); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').close
 prices[os.path.basename(p)[:-4]]=d
px=pd.DataFrame(prices).sort_index()
# defensive low volatility: negative ratio recent 5d realized / long 30d realized
r=np.log(px).diff(); rv5=r.rolling(5).std(); rv30=r.rolling(30).std()
f=-(rv5/rv30)
# forward one-day return
fr=px.pct_change().shift(-1)
ics=[]; rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): ics.append(ic); rows.append((dt,ic,len(z)))
a=np.array(ics); print('cutoff',cut,'dates',len(a),'mean_ic',a.mean(),'icir',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'mean_n',np.mean([x[2] for x in rows]))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-02-24')]:
 q=[v for d,v,n in rows if str(d.date())>=lo and str(d.date())<=hi]
 print(lo,hi,len(q),round(np.mean(q),5) if q else None,round(np.mean(q)/np.std(q,ddof=1),5) if len(q)>1 else None)
# artifact
out=pd.DataFrame(f).reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('../persistent/factor_signals_miner_2_20270226_volratio5_30.csv',index=False)
