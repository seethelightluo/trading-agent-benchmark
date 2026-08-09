import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2027-02-25'
xs={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
 d=d[d.date<=cut].set_index('date')
 # gap from prior close to today's open; factor contrarian
 gap=d.open/d.close.shift(1)-1
 # forward close-close return
 fwd=d.close.pct_change().shift(-1)
 xs[s]=pd.DataFrame({'f':-gap,'r':fwd})
rows=[]
for dt in sorted(set().union(*[x.index for x in xs.values()])):
 vals=[]; rets=[]
 for s,x in xs.items():
  if dt in x.index and np.isfinite(x.loc[dt,'f']) and np.isfinite(x.loc[dt,'r']): vals.append(x.loc[dt,'f']);rets.append(x.loc[dt,'r'])
 if len(vals)>=8: rows.append((dt,spearmanr(vals,rets).statistic,len(vals)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for h in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 q=z.loc[h[0]:h[1],'ic'];print(h,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# 5d/10d forward close returns
for k in [1,5,10]:
 rr=[]
 for s in U:
  d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date');d=d[d.date<=cut].set_index('date'); f=-d.open/d.close.shift(1)+1; r=d.close.pct_change(k).shift(-k+1) # return t to t+k-1? 
  rr.append(pd.DataFrame({'f':f,'r':r,'s':s}))
 all=pd.concat(rr)
 out=[]
 for dt,g in all.groupby(level=0):
  g=g.replace([np.inf,-np.inf],np.nan).dropna()
  if len(g)>=8:out.append(spearmanr(g.f,g.r).statistic)
 print('h',k,'IC',np.nanmean(out),'ICIR',np.nanmean(out)/np.nanstd(out,ddof=1),'n',len(out))
# signal artifact
sig=[]
for s,x in xs.items():
 y=x[['f']].rename(columns={'f':s});sig.append(y)
pd.concat(sig,axis=1).to_csv('../persistent/factor_signals_miner_1_20270225_gap_reversal.csv')
