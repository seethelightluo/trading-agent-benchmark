import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-05-31')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut] for a in assets}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]
C=pd.DataFrame(px).sort_index(); r=C.pct_change(); vr=v.reindex(C.index).ffill().pct_change()
beta=r.rolling(60,min_periods=45).cov(vr).div(vr.rolling(60,min_periods=45).var(),axis=0)
# Negative VIX beta, scaled by observable VIX stress relative to its trailing median.
state=(v.reindex(C.index).ffill()/v.reindex(C.index).ffill().rolling(120,min_periods=80).median()).clip(0.5,2.0)
factor=-beta.mul(state,axis=0).shift(1)
def run(h):
 fwd=C.pct_change(h).shift(-h); qs=[]; ds=[]; ns=[]
 for d in C.index:
  z=pd.concat([factor.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8: qs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
 q=pd.Series(qs,index=ds); return q,ns
q,ns=run(1)
print('cutoff',cut.date(),'dates',len(q),'mean_names',round(float(np.mean(ns)),2),'coverage',round(float(factor.stack().notna().mean()),4))
print('1d IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('years',q.groupby(q.index.year).mean().round(5).to_dict())
for h in [5,10,20]:
 x,n=run(h);print('%dd IC %.6f ICIR %.6f dates %d'%(h,x.mean(),x.mean()/x.std(ddof=1),len(x)))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
