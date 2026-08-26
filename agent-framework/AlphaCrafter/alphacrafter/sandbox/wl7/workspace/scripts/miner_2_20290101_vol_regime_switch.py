import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-12-31')
p={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 p[a]=d[d.index<=cut]
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill()
vchg=v.pct_change(10)
ret20=p/p.shift(20)-1; vol20=r.rolling(20).std()*np.sqrt(252)
trend=ret20/vol20.replace(0,np.nan); rev=-trend
sig=trend.where(vchg<=0,rev).shift(1); fwd=p.shift(-10)/p-1
rows=[]
for dt in p.index:
 x,y=sig.loc[dt],fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); rank=sig.rank(axis=1,pct=True); to=rank.diff().abs().mean(axis=1).reindex(z.index)
print('candidate=vol_regime_switch_trend_reversal'); print('dates',len(z),'avg_n',z.n.mean(),'period',z.index.min().date(),z.index.max().date())
print('mean_ic %.6f icir %.6f hit %.4f turnover %.6f coverage %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean(),to.mean(),z.n.mean()/15))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
 q=z.loc[lo:hi,'ic']; print(lo+'-'+hi,'n',len(q),'ic',round(q.mean(),6) if len(q) else np.nan,'icir',round(q.mean()/q.std(),4) if len(q)>1 else np.nan,'hit',round((q>0).mean(),3) if len(q) else np.nan)
sig.to_csv('scripts/miner_2_20290101_vol_regime_switch_signal.csv',index_label='date')
