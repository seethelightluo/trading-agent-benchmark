import pandas as pd, numpy as np, os, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2026-10-21')
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.sort_values('date').set_index('date'); v=v.loc[:cutoff]; vix=v.close.pct_change()
series={}
for a in assets:
 p=f'{base}/{a}.csv'
 if not os.path.exists(p): continue
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date').loc[:cutoff]
 shock=vix.reindex(d.index).shift(1); ret=d.close.pct_change(); vol=ret.rolling(20,min_periods=15).std().shift(1)
 # prior VIX-up shock gates a volatility-scaled fade of prior asset return
 fac=(-ret/vol)*((shock>0).astype(float)); series[a]=pd.DataFrame({'f':fac})
def calc(h):
 rows=[]
 for a,x in series.items():
  d=pd.read_csv(f'{base}/{a}.csv'); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date').loc[:cutoff]
  q=x.copy(); q['r']=d.close.shift(-h)/d.close-1; q['a']=a; rows.append(q.reset_index())
 dd=pd.concat(rows,ignore_index=True).dropna(subset=['f','r']); obs=[]
 for dt,g in dd.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   ic=spearmanr(g.f,g.r).statistic
   if np.isfinite(ic): obs.append((dt,ic,len(g)))
 z=pd.DataFrame(obs,columns=['date','ic','n'])
 return z
for h in [1,5,10]:
 z=calc(h); ic=z.ic.mean(); ir=ic/z.ic.std(ddof=1)
 print(h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.sum()/(len(z)*15),4),'IC',round(ic,5),'ICIR',round(ir,5),'hit',round(np.mean(z.ic>0),4))
z=calc(1); print('period',z.date.min().date(),z.date.max().date())