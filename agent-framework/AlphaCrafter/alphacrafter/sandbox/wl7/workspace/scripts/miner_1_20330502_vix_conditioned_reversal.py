import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
r=p.pct_change(); ret10=p/p.shift(10)-1; vol30=r.rolling(30).std()*np.sqrt(10)
x=vix.pct_change(5); vz=(x-x.rolling(60).mean())/x.rolling(60).std(); stress=(1+0.5*vz.clip(-2,2)).clip(0.25,2.0)
sig=-(ret10/vol30).mul(stress,axis=0).shift(1); fwd=p.shift(-10)/p-1
rows=[]
for d in sig.index:
 a=sig.loc[d]; y=fwd.loc[d]; ok=a.notna()&y.notna()
 if ok.sum()>=8: rows.append((d,spearmanr(a[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=z[z.index>=pd.Timestamp('2026-07-16')]
print('dates',len(z),'assets',z.n.mean(),'coverage',sig.loc[z.index].notna().mean().mean())
print('IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
parts=np.array_split(np.arange(len(z)),3); print('thirds',[z.iloc[q]['ic'].mean() for q in parts])
rank=sig.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean())
out=sig.loc[z.index].stack().rename('signal').reset_index(); out.columns=['date','asset','signal']; out.to_csv('scripts/miner_1_20330502_vix_conditioned_reversal_signal.csv',index=False)
