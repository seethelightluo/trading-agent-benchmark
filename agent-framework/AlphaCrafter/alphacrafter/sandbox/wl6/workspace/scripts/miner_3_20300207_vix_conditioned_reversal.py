import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2030-02-07')
base='../persistent/stock_data'; macro=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]) & set(macro.index)); dates=[d for d in dates if d<=cutoff]; out=[]
for d in dates:
 ix=dates.index(d)
 if ix<30 or ix+10>=len(dates): continue
 vals=[]; fw=[]; vv=macro.reindex(dates[:ix+1]).iloc[-20:]
 if pd.isna(macro.loc[d]) or len(vv)<20: continue
 vz=(macro.loc[d]-vv.mean())/(vv.std()+1e-12)
 for s in U:
  q=px[s].reindex(dates[:ix+1]).dropna()
  if len(q)<30: continue
  r5=q.iloc[-1]/q.iloc[-6]-1; vol=q.pct_change().iloc[-20:].std()*np.sqrt(20)+1e-9
  vals.append((s,-r5/vol*(1+0.35*np.tanh(vz)))); qf=px[s].reindex(dates); fw.append((s,qf.loc[d]/qf.iloc[ix+10]-1))
 a=dict(vals); b=dict(fw); common=[s for s in a if s in b and np.isfinite(a[s]) and np.isfinite(b[s])]
 if len(common)>=8: out.append((d,spearmanr([a[s] for s in common],[b[s] for s in common]).statistic,len(common)))
x=pd.DataFrame(out,columns=['date','ic','n']); print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.sum()/(len(x)*15)); print('IC',x.ic.mean(),'ICIR',x.ic.mean()/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x.ic>0).mean()); print(x.assign(year=pd.to_datetime(x.date).dt.year).groupby('year').ic.mean().to_string())
