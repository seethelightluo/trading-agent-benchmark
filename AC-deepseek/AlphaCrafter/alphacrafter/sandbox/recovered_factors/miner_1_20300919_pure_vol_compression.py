import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame(px).sort_index().astype(float); r=p.pct_change()
# Pure lagged volatility compression: lower short volatility relative to long volatility.
v5=r.rolling(5,min_periods=4).std().shift(1); v40=r.rolling(40,min_periods=25).std().shift(1)
sig=(-(v5/v40)).replace([np.inf,-np.inf],np.nan)
print('candidate=lagged_pure_vol_compression_5_40'); print('dates',len(p),'instruments',len(p.columns),'coverage',round(sig.notna().sum().sum()/sig.size,6),'meanN',round(sig.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 a=[];ns=[]; f=p.shift(-h)/p-1
 for d in p.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a);print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [10,20]:
 f=p.shift(-h)/p-1
 for label,mask in [('2024-27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('2028+',p.index>='2028-01-01'),('latest120',p.index>=p.index[-120])]:
  a=[]
  for d in p.index[mask]:
   z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.asarray(a);print('regime',h,label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
for lag in [1,5,10,20]:
 z=pd.concat([sig.stack().rename('a'),sig.shift(-lag).stack().rename('b')],axis=1).dropna();print('signal_decay',lag,round(spearmanr(z.a,z.b).statistic,6),len(z))
# reconstructed admitted comparators, explicit max abs correlation
comps={}
comps['ravmom_20obs']=r.rolling(20).sum().shift(1)
comps['risk_adjusted_trend_20d']=comps['ravmom_20obs']/r.rolling(20).std().shift(1)
comps['trend_acceleration']=r.rolling(5).sum().shift(1)-r.rolling(20).sum().shift(1)
comps['volnorm_reversal_5obs']=-r.rolling(5).sum().shift(1)/r.rolling(20).std().shift(1)
comps['recovery_acceleration']=p.pct_change(5).shift(1)-p.pct_change(20).shift(1)
comps['inverse_expected_shortfall']=-(r.rolling(20).apply(lambda x: np.mean(x[x<=np.quantile(x,.2)]) if len(x) else np.nan)).shift(1)
comps['trend_consistency']=np.sign(r).rolling(20).mean().shift(1)
comps['inverse_vol']=(-r.rolling(20).std()).shift(1)
comps['inverse_excess_kurtosis']=(-r.rolling(60).kurt()).shift(1)
mx=0;who=''
for k,c in comps.items():
 z=pd.concat([sig.stack().rename('a'),c.stack().rename('b')],axis=1).dropna(); q=abs(spearmanr(z.a,z.b).statistic)
 print('library_corr',k,round(q,6),len(z))
 if q>mx:mx=q;who=k
print('max_abs_library_correlation',round(mx,6),who)
