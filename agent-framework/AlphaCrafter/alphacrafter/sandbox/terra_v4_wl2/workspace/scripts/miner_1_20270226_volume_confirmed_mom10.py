import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 D[s]=d
# Idea: volume-confirmed medium momentum; log volume surprise times 10d return, all lagged naturally
rows=[]
for s,d in D.items():
 c=d.close.astype(float); v=d.volume.astype(float).replace(0,np.nan)
 mom=c.pct_change(10)
 vs=np.log(v/(v.rolling(60,min_periods=30).median()))
 f=mom*vs.clip(-3,3)
 f=f.replace([np.inf,-np.inf],np.nan)
 fr=c.shift(-5)/c-1
 for dt in f.index:
  rows.append((dt,s,f.loc[dt],fr.loc[dt]))
x=pd.DataFrame(rows,columns=['date','sym','f','r']).dropna()
ics=[]; sig=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
  ics.append(spearmanr(g.f,g.r).statistic); sig.append((dt,g))
ics=np.array(ics); print('dates',len(ics),'avgN',np.mean([len(g) for _,g in sig]),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean())
for name,lo,hi in [('all','2020','2027-12'),('pre','2020','2026-07-16'),('online','2026-07-16','2027-02-26'),('recent','2026-01-01','2027-02-26')]:
 a=np.array([z for (dt,_),z in zip(sig,ics) if str(dt)>=lo and str(dt)<hi]); print(name,len(a), a.mean() if len(a) else np.nan, (a.mean()/a.std(ddof=1)) if len(a)>1 else np.nan)
# turnover ranks
wide=x.pivot(index='date',columns='sym',values='f'); ranks=wide.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna().mean(); print('coverage',len(x)/(len(D)*len(set(x.date))),'turnover',turn)
# artifact
out=x.pivot(index='date',columns='sym',values='f'); out.to_csv('../persistent/factor_signals_miner_1_20270226_volume_confirmed_mom10.csv')
