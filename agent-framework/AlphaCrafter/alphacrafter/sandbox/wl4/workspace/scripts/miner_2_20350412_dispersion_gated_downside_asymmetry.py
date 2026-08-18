import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in S],axis=1).sort_index(); r=p.pct_change()
neg=r.clip(upper=0); down=neg.rolling(40,min_periods=25).std(); total=r.rolling(20,min_periods=15).sum()
base=(-total/down.replace(0,np.nan)).shift(1)
# Use only signals when lagged cross-asset dispersion is above its trailing median.
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
gate=(disp.shift(1)>disp.shift(1).rolling(120,min_periods=60).median()).astype(float)
sig=base.mul(gate,axis=0).replace(0,np.nan); os.makedirs('scripts/artifacts',exist_ok=True)
rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],(p.shift(-10)/p-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=dispersion_gated_downside_asymmetry dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15));print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for k in [120,260,520,1040]:
 q=a.tail(k); print('recent',k,'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
print('gate_fraction',gate.mean(),'turnover_proxy',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; x=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'IC %.8f ICIR %.8f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1)))
sig.to_csv('scripts/artifacts/miner_2_20350412_dispersion_gated_downside_asymmetry_signal.csv');a.to_csv('scripts/artifacts/miner_2_20350412_dispersion_gated_downside_asymmetry_ic.csv')
