import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-12-12']
r=P.pct_change(); bench=r.mean(axis=1)
# Equal-weight benchmark beta residual reversal; all rolling inputs lagged one day.
vol=r.rolling(20,min_periods=15).std().shift(1)
beta=r.rolling(60,min_periods=40).cov(bench).div(bench.rolling(60,min_periods=40).var(),axis=0).shift(1)
res=r.sub(beta.mul(bench,axis=0),axis=0)
# Blend 5d and 10d residual reversal, volatility normalized.
f=-(0.6*res.rolling(5,min_periods=4).sum()+0.4*res.rolling(10,min_periods=7).sum())/vol
rows={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; out=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 R=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); x=R.ic; rows[h]=R
 print('h',h,'dates',len(x),'avgN',round(R.n.mean(),2),'coverage',round(R.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for nm,q in [('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('r360',slice('2028-12-12','2029-12-12')),('r180',slice('2029-06-12','2029-12-12'))]:
  y=R.loc[q,'ic']; print(nm,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
rows[10].to_csv('scripts/miner_2_20291213_residual_blend_ic.csv')
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20291213_residual_blend_signal.csv',index=False)
