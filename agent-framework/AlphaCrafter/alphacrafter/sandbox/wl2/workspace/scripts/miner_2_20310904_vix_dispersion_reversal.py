import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr

symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in symbols:
    p=f'../persistent/stock_data/{s}.csv'
    d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
    prices[s]=d['close'].replace(0,np.nan)
px=pd.DataFrame(prices).sort_index()
ret=px.pct_change()
# Cross-sectional 5d dispersion and its trailing 252-day median, plus VIX 5d increase
r5=px.pct_change(5)
disp=r5.std(axis=1)
threshold=disp.rolling(252,min_periods=126).median()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date')['close']
vix5=vix.reindex(px.index).ffill().pct_change(5)
active=(disp>threshold)&(vix5>0)
sig=(-r5).where(active,0.0)
fwd=ret.shift(-1)
rows=[]
for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z),active.loc[dt]))
out=pd.DataFrame(rows,columns=['date','ic','n','active']).set_index('date')
# exclude latest incomplete forward observation
out=out.iloc[:-1]
print('dates',len(out),'avg_n',out.n.mean(),'coverage',out.n.sum()/(len(out)*15),'active_rate',out.active.mean())
print('IC %.9f ICIR %.9f hit %.4f turnover %.4f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1), (out.ic>0).mean(), sig.diff().abs().sum(axis=1).mean()))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2031-09-03')]:
 q=out.loc[a:b].ic; print(a[:4]+'-'+b[:4],len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
os.makedirs('scripts',exist_ok=True)
sig.stack().rename('signal').to_csv('scripts/miner_2_20310904_vix_dispersion_reversal_signal.csv',header=True)
