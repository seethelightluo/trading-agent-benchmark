import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data'); macro=Path('../persistent/index_data')
def load(s):
 d=pd.read_csv(s); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').sort_index()
d={s:load(base/(s+'.csv')) for s in U}; v=load(macro/'VIX.csv').close.astype(float); shock=v.pct_change(fill_method=None).rolling(5,min_periods=5).sum().clip(lower=0)
f={}; fw={}
for s,x in d.items():
 # signal uses today's completed OHLC; forward is next observation close return
 f[s]=(-(x.close/x.open-1)*(1+shock.reindex(x.index).fillna(0))).rename(s)
 fw[s]=(x.close.shift(-1)/x.close-1).rename(s)
f=pd.DataFrame(f).sort_index(); fw=pd.DataFrame(fw).sort_index(); rows=[]
for dt in f.index:
 a=f.loc[dt].dropna(); b=fw.loc[dt].reindex(a.index).dropna(); a=a.reindex(b.index)
 if len(a)>=8: rows.append((dt,spearmanr(a,b).statistic,len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15)); print('IC %.5f ICIR %.5f hit %.3f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean()))
for name,st,en in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31')]:
 z=r.loc[st:en].ic; print(name,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20261119_vix_intraday_reversal_signal.csv')
