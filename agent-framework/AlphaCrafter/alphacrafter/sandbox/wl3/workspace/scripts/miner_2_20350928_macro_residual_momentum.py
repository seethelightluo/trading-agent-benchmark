import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2035-09-27')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in syms:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]
 px[s]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut].reindex(p.index).ffill().pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut].reindex(p.index).ffill().pct_change()
# Macro-residual medium momentum: 20d return stripped of rolling beta to DXY and VIX shocks.
rd=r.rolling(20).sum(); x=pd.concat([dxy,vix],axis=1); x.columns=['d','v'];
# rolling covariance beta, aligned using information through t
f=pd.DataFrame(index=p.index,columns=syms,dtype=float)
for s in syms:
 y=r[s]
 bd=y.rolling(60).cov(dxy)/dxy.rolling(60).var()
 bv=y.rolling(60).cov(vix)/vix.rolling(60).var()
 f[s]=rd[s]-bd*(dxy.rolling(20).sum())-bv*(vix.rolling(20).sum())
# IC at horizons and stability
out=[]
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(p)-h):
  a=f.iloc[i]; b=p.pct_change(h).shift(-h).iloc[i]
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 out.append((h,len(vals),np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1),np.mean(np.array(vals)>0)))
# turnover rank changes
rank=f.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).mean()
print('cutoff',cut.date(),'dates',len(p),'avg instruments',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean())
print('horizon n IC ICIR hit')
for x in out: print(x)
print('turnover',turnover)
for name,lo,hi in [('early',0,len(p)//3),('middle',len(p)//3,2*len(p)//3),('recent',2*len(p)//3,len(p))]:
 vals=[]
 for i in range(lo,min(hi,len(p)-10)):
  z=pd.concat([f.iloc[i],p.pct_change(10).shift(-10).iloc[i]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(name,len(vals),np.nanmean(vals))
# artifact
f.to_csv('scripts/miner_2_20350928_macro_residual_momentum_signal.csv',index_label='date')
