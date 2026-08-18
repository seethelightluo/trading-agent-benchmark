import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-01-08'); base='../persistent/stock_data'
p={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).query('date <= @cut').set_index('date')['close'] for s in U}
p=pd.DataFrame(p).sort_index(); ret=p.pct_change(); r10=p.pct_change(10); resid=r10.sub(r10.median(axis=1),axis=0)
vol=ret.rolling(40,min_periods=20).std(); sig=(-resid.rolling(3,min_periods=3).mean()/(vol+1e-8)).shift(1)
print('candidate=volatility_normalized_residual_reversal; cutoff',cut.date(),'dates',len(p),'instruments',len(U))
for h in [5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]; dates=[]
 for d in p.index:
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z)); dates.append(d)
 a=np.asarray(vals); ic=a.mean(); icir=ic/(a.std(ddof=1)/np.sqrt(len(a)))
 print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={ic:.8f} ICIR={icir:.8f} hit={(a>0).mean():.6f} coverage={np.mean(ns)/15:.6f}')
 if h==10:
  for n in [365,730,1095]:
   b=a[-n:]; print(f'recent_{n} IC={b.mean():.8f} ICIR={b.mean()/(b.std(ddof=1)/np.sqrt(len(b))):.8f}')
print('rank_turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
