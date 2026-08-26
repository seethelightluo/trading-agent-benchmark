import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'))
 d['date']=pd.to_datetime(d['date']); d=d.set_index('date')['close'].astype(float)
 px[s]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# medium trend normalized by realized risk; market-wide dispersion gates trend in coherent regimes
raw=p.pct_change(20)/(r.rolling(40).std()*np.sqrt(20))
disp=r.rolling(20).std().mean(axis=1)
# lower dispersion = more coherent; smoothly reward trend, lagged
z=(disp-disp.rolling(120).median())/(disp.rolling(120).median()+1e-12)
factor=raw.mul((1-0.35*np.tanh(z)),axis=0).shift(1)
for h in [10,20,40,60]:
 fw=p.shift(-h)/p-1
 vals=[]; dates=[]; ns=[]
 for dt in factor.index:
  a=factor.loc[dt]; b=fw.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(a[ok],b[ok]).statistic); dates.append(dt); ns.append(ok.sum())
 x=pd.Series(vals,index=dates).dropna(); print(h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),len(x),np.mean(ns)))
# coverage and rank turnover
print('coverage',factor.notna().mean().mean(),'turnover',factor.rank(axis=1,pct=True).diff().abs().mean().mean())
# regimes
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 fw=p.shift(-60)/p-1; q=[]
 for dt in factor.loc[a:b].index:
  x=factor.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print(a,b,'regime60',np.nanmean(q),'n',len(q))
# signal artifact
out=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20340316_dispersion_conditioned_trend_signal.csv',index=False)
