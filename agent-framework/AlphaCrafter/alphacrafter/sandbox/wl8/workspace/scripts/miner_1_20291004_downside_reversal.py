import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index().loc[:'2029-10-03']; r=P.pct_change()
# downside-asymmetry reversal: prior 10d return, normalized by downside deviation over prior 30d
neg=r.clip(upper=0); dd=neg.rolling(30,min_periods=15).std()*np.sqrt(30)
sig=(-P.pct_change(10)/dd).shift(1)
def ev(h):
 f=P.shift(-h)/P-1; z=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append((d,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
 return pd.DataFrame(z,columns=['date','n','ic']).set_index('date')
q=ev(10);print('period',q.index.min().date(),q.index.max().date(),'dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15));print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,s in [('2026',q.loc['2026']),('2027-28',q.loc['2027':'2028']),('recent360',q.tail(360)),('recent180',q.tail(180))]:print(name,len(s),s.ic.mean(),s.ic.mean()/s.ic.std(ddof=1))
for h in [1,5,10,20]:
 a=ev(h).ic.to_numpy();print('decay',h,np.mean(a),np.mean(a)/np.std(a,ddof=1),len(a))
sig.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20291004_downside_reversal_signal.csv',index=False)
