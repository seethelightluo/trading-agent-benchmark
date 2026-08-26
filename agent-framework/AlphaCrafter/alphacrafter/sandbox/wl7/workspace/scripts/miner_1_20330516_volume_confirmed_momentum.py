import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
frames={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date') for a in assets}
close=pd.DataFrame({a:frames[a]['close'] for a in assets}).sort_index()
vol=pd.DataFrame({a:frames[a]['volume'] for a in assets}).reindex(close.index)
r=close.pct_change()
# Volume-confirmed trend: medium momentum, scaled by abnormal volume, with all information lagged.
ret20=close/close.shift(20)-1
rv40=r.rolling(40).std()*np.sqrt(10)
volratio=vol/vol.rolling(40).median()
# winsorize volume confirmation to prevent one-off distortions
confirm=volratio.clip(0.5,2.0).pow(0.5)
sig=(ret20/rv40*confirm).shift(1)
fwd=close.shift(-10)/close-1
rows=[]
for d in sig.index:
    a,y=sig.loc[d],fwd.loc[d]; ok=a.notna()&y.notna()
    if ok.sum()>=8: rows.append((d,spearmanr(a[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=z[z.index>=pd.Timestamp('2026-07-16')]
print('validation_start 2026-07-16 end',z.index.max().date(),'dates',len(z),'avg_assets',z.n.mean())
print('coverage',sig.loc[z.index].notna().mean().mean())
print('IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
print('horizons')
for h in [1,5,10,20]:
 f=close.shift(-h)/close-1; rr=[]
 for d in sig.index:
  a,y=sig.loc[d],f.loc[d]; ok=a.notna()&y.notna()
  if ok.sum()>=8: rr.append(spearmanr(a[ok],y[ok]).statistic)
 rr=np.asarray(rr)[np.isfinite(rr)]; rr=rr[len(rr)-len(z):]
 print(h,rr.mean(),rr.mean()/rr.std(ddof=1))
rank=sig.rank(axis=1,pct=True)
print('turnover',rank.diff().abs().mean().mean())
third=np.array_split(np.arange(len(z)),3); print('thirds',[z.iloc[i].ic.mean() for i in third])
out=sig.loc[z.index].stack().rename('signal').reset_index(); out.columns=['date','asset','signal']; out.to_csv('scripts/miner_1_20330516_volume_confirmed_momentum_signal.csv',index=False)
