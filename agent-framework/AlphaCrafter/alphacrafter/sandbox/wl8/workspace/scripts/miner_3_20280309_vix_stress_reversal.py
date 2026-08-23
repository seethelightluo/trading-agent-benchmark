import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-03-08')
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d[d.index<=end]
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); v=v[v.index<=end].reindex(P.index).ffill()
# signal on d uses through d-1, lagged 3d reversal, VIX level z through d-1
vr=v.pct_change(1); vz=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std()
base=-(P.shift(1)/P.shift(4)-1)
# stress amplification, clipped positive z; factor remains interpretable
f=base.mul((1+0.75*vz.shift(1).clip(lower=0,upper=2)), axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
# target next close return from d to d+1
fw=P.shift(-1)/P-1
rows=[]
for d in P.index:
 x=f.loc[d]; y=fw.loc[d]
 ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((d,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=VIX-stress-amplified 3d reversal')
print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*15))
print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
for name,mask in [('2020-22',(z.index<'2023-01-01')),('2023-25',((z.index>='2023-01-01')&(z.index<'2026-01-01'))),('2026',((z.index>='2026-01-01')&(z.index<'2027-01-01'))),('2027+',z.index>='2027-01-01'),('recent90',z.index>=end-pd.Timedelta(days=90))]:
 q=z[mask]; print(name,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std()) if len(q) else '')
# horizons via forward aggregated, non-overlapping daily obs still
for h in [2,5,10]:
 target=P.shift(-h)/P-1; rr=[]
 for d in P.index:
  x=f.loc[d]; y=target.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(spearmanr(x[ok],y[ok]).statistic)
 q=pd.Series(rr).dropna(); print('h',h,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
# signal artifact
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20280309_vix_stress_reversal_signal.csv',index=False)
