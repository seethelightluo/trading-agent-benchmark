import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index().loc[:'2029-11-14']
r=P.pct_change(); m=r.mean(axis=1)
# Reversal of each asset's 10d move relative to the equal-weight cross-asset move,
# scaled by lagged idiosyncratic 20d volatility. Entire signal is lagged one day.
res=r.sub(m,axis=0)
vol=res.rolling(20,min_periods=15).std().shift(1)
sig=(-res.rolling(10,min_periods=8).sum().shift(1)/vol).replace([np.inf,-np.inf],np.nan)
def ev(h):
 f=P.shift(-h)/P-1; rows=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   rows.append((d,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 return pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
q=ev(10); print('period',q.index.min().date(),q.index.max().date(),'dates',len(q),'avg_n',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4)); print('IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
for nm,sub in [('2026',q.loc['2026']),('2027-28',q.loc['2027':'2028']),('recent360',q.tail(360)),('recent180',q.tail(180))]: print(nm,len(sub),round(sub.ic.mean(),6),round(sub.ic.mean()/sub.ic.std(ddof=1),6))
for h in [5,10,20]:
 a=ev(h).ic; print('decay',h,round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),len(a))
print('turnover',round(float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
sig.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20291115_residual_vol_reversal_signal.csv',index=False)
