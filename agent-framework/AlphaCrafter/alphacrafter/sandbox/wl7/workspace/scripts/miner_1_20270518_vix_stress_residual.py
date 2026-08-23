import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
wide=pd.DataFrame({a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}).sort_index(); rets=wide.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(wide.index).ffill(); vp=vix.shift(1).rolling(252,min_periods=60).rank(pct=True).clip(0,1)
r3=wide.pct_change(3).shift(1); vol=rets.shift(1).rolling(20,min_periods=15).std(); res=r3.sub(r3.mean(axis=1),axis='index')
f=(-res/vol*(0.5+vp.fillna(.5))).replace([np.inf,-np.inf],np.nan); fwd=wide.pct_change().shift(-1)
rows=[]
for d in f.index:
 ok=f.loc[d].notna()&fwd.loc[d].notna()
 if ok.sum()>=8: rows.append((d,spearmanr(f.loc[d][ok],fwd.loc[d][ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').sort_index(); print('span',wide.index.min().date(),wide.index.max().date(),'dates',len(z))
for end in [z.index.max(),pd.Timestamp('2027-05-18')]:
 q=z.loc[:end]
 if len(q):
  print('END',end.date(),'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5),'hit',round((q.ic>0).mean(),4),'coverage',round(q.n.mean()/15,4))
  for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-05-18')]:
   a=q.loc[lo:hi]; print('REG',lo,len(a),round(a.ic.mean(),5) if len(a) else np.nan)
for h in [1,5,10]:
 yy=wide.pct_change(h).shift(-h); rr=[]
 for d in f.index:
  ok=f.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8: rr.append(spearmanr(f.loc[d][ok],yy.loc[d][ok]).statistic)
 print('DECAY',h,round(np.nanmean(rr),5),len(rr))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('scripts/miner_1_20270518_vix_stress_residual_signal.csv',index=False)
