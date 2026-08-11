import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b=Path('../persistent/stock_data'); p={}
for s in syms:
 d=pd.read_csv(b/f'{s}.csv'); d['date']=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(p).sort_index().ffill(); r=np.log(p).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float).reindex(p.index).ffill()
# Volatility-conditioned short-term reversal: fade 5d moves only when VIX is elevated vs its 60d median.
rv=r.rolling(20).std(); shock=(v>v.rolling(60).median()).astype(float)
f=(-r.rolling(5).sum()/rv).mul(shock,axis=0).replace(0,np.nan).shift(1)
for h in [5,10,20]:
 fw=np.log(p.shift(-h)/p); z=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(q)>=8: z.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
 x=pd.DataFrame(z,columns=['date','ic','n']).set_index('date')
 print(f'H={h} dates={len(x)} avg_n={x.n.mean():.2f} IC={x.ic.mean():.6f} ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f} hit={(x.ic>0).mean():.4f}')
 for lab,cut in [('2026+','2026-01-01'),('2027+','2027-01-01'),('2028+','2028-01-01')]:
  y=x[x.index>=cut]; print(f' {lab} n={len(y)} IC={y.ic.mean():.6f} ICIR={y.ic.mean()/y.ic.std(ddof=1):.6f}')
rank=f.rank(axis=1,pct=True); print(f'panel_dates={len(f)} instruments={len(syms)} coverage={f.notna().mean().mean():.6f} rank_turnover={(rank-rank.shift()).abs().mean(axis=1).dropna().mean():.6f}')
f.reset_index().to_csv('scripts/miner_1_20280810_vix_reversal_signal.csv',index=False)
