import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); px[a]=d.set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); rr=r.rolling(5).sum(); peer=rr.sub(rr.median(axis=1),axis=0)
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.reindex(p.index).ffill()
broad=rr.median(axis=1); low=(v<v.rolling(252,min_periods=126).median()).astype(float)
f=peer.where((broad>0)&(low>0)).shift(1); fwd=p.shift(-5).div(p)-1
ics=[]; dates=[]; inst=[]; cov=[]; turns=[]; prev=None
for dt in f.index:
 x=f.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); inst.append(ok.sum()); cov.append(ok.mean())
  rank=x.rank(pct=True); turns.append((rank-prev).abs().mean() if prev is not None else np.nan); prev=rank
z=pd.Series(ics,index=pd.to_datetime(dates)).dropna(); t=pd.Series(turns).dropna()
print('dates',len(z),'avg_inst',np.mean(inst),'universe',15)
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f active %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean(),np.mean(cov),t.mean(),(f.notna().sum(axis=1)>0).mean()))
for name,lo,hi in [('2020-22','2020-01-01','2023-01-01'),('2023-25','2023-01-01','2026-01-01'),('2026-31','2026-01-01','2032-01-01')]:
 zz=z[(z.index>=lo)&(z.index<hi)]; print(name,len(zz), 'IC %.6f ICIR %.6f'%(zz.mean(),zz.mean()/zz.std() if len(zz)>1 else np.nan))
f.index.name='date'; f.to_csv('scripts/miner_1_20310626_conditional_peer_momentum_signal.csv')
