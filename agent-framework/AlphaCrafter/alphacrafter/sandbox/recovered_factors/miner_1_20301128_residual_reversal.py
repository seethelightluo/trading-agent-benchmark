import pandas as pd, numpy as np, glob, os, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
# candidate: negative 20d residual momentum, residualized cross-sectionally to equal-weight market return
mkt=ret.mean(axis=1); mom=prices/prices.shift(20)-1; mret=mkt.rolling(20).sum()
res=mom.sub(mret,axis=0); fac=-res
# signal date t predicts close return t+10, require all assets
fwd=prices.shift(-10)/prices-1
rows=[]
for d in fac.index:
 x=fac.loc[d]; y=fwd.loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((d,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate residual reversal H10')
print('dates',len(r),'meanN',r.n.mean(),'IC %.8f ICIR %.8f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31')]:
 q=r.loc[lo:hi].ic; print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for n in [1,5,10,20]:
 rr=[]
 ff=prices/prices.shift(20)-1
 yy=prices.shift(-n)/prices-1
 for d in fac.index:
  x=fac.loc[d]; y=yy.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(spearmanr(x[ok],y[ok]).statistic)
 z=pd.Series(rr).dropna(); print('H',n,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
# turnover and coverage
rank=fac.rank(axis=1,pct=True); print('coverage',fac.notna().mean().mean(),'turnover',rank.diff().abs().mean().mean())
# reconstruct broad library proxies and max corr; use date-cell pooled rank correlations
proxies={'mom20':mom,'reversal5':-(prices/prices.shift(5)-1),'trendcons':(prices/prices.shift(5)-1)+(prices/prices.shift(20)-1),'volnormrev':-(prices/prices.shift(5)-1)/ret.rolling(20).std()}
for k,v in proxies.items():
 z=pd.concat([fac.stack(),v.stack()],axis=1).dropna(); print('corr',k,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
# macro conditional residual reversal: candidate unchanged, report latest
print('latest120',r.tail(120).ic.mean(),r.tail(120).ic.mean()/r.tail(120).ic.std(ddof=1))
