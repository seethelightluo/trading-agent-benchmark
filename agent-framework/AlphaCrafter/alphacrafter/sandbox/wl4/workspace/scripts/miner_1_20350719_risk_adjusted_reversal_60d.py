import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
P=pd.concat(D,axis=1).sort_index(); R=P.pct_change(); L=60; H=10
sig=-(P.pct_change(L)/(R.rolling(L).std()*np.sqrt(L))).shift(1); fwd=P.shift(-H)/P-1
rows=[]; prev=None; turns=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,x[ok].corr(y[ok]),int(ok.sum())))
  z=x[ok].rank(pct=True)
  if prev is not None:
   common=z.index.intersection(prev.index); turns.append(np.mean(abs(z[common]-prev[common])))
  prev=z
A=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=A.ic.dropna()
print('universe',len(U),'dates',len(q),'avgN',A.n.mean(),'coverage',sig.notna().mean().mean())
print('IC %.8f ICIR %.8f hit %.4f turnover %.6f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),np.mean(turns)))
for n in [120,260,520,1040]:
 z=q.tail(n); print('recent',n,'dates',len(z),'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('decay',end=' ')
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; a=[]
 for dt in sig.index:
  x=sig.loc[dt]; v=y.loc[dt]; ok=x.notna()&v.notna()
  if ok.sum()>=8:a.append(x[ok].corr(v[ok]))
 print(h,round(np.nanmean(a),8),end='; ')
print()
A.to_csv('scripts/artifacts/miner_1_20350719_risk_adjusted_reversal_60d_ic.csv'); sig.to_csv('scripts/artifacts/miner_1_20350719_risk_adjusted_reversal_60d_signal.csv')
