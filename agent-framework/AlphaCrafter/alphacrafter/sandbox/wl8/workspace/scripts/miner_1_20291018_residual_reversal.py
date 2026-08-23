import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index().loc[:'2029-10-16']
r=P.pct_change(); market=r.mean(axis=1)
# Idiosyncratic 5-day move versus contemporaneous equal-weight benchmark; lagged before ranking.
sig=-(r.rolling(5,min_periods=5).sum().sub(market.rolling(5,min_periods=5).sum(),axis=0)).shift(1)
def ev(h):
 fwd=P.shift(-h)/P-1; out=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: out.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 return pd.DataFrame(out,columns=['date','n','ic']).set_index('date')
q=ev(10); print('period',q.index.min().date(),q.index.max().date(),'dates',len(q),'avg_n',round(q.n.mean(),3),'coverage',round(q.n.sum()/(len(q)*15),4));print('IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4));
for name,sub in [('2026',q.loc['2026']),('2027-28',q.loc['2027':'2028']),('recent360',q.tail(360)),('recent180',q.tail(180))]: print(name,len(sub),round(sub.ic.mean(),6),round(sub.ic.mean()/sub.ic.std(ddof=1),6))
for h in [1,5,10,20]:
 a=ev(h).ic;print('decay',h,round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),len(a))
print('turnover_proxy',round(float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))
sig.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20291018_residual_reversal_signal.csv',index=False)
