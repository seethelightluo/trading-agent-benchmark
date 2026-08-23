import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-11-28']
r=P.pct_change()
# Reversal of acceleration: assets whose recent 10d return deteriorated versus preceding 10d,
# normalized by lagged 20d volatility. All inputs are known at t-1.
recent=P.pct_change(10).shift(1)
prior=P.pct_change(20).shift(1)-recent
vol=r.rolling(20,min_periods=15).std().shift(1)
f=-(recent-prior)/vol.replace(0,np.nan)
rows_by={}
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic; rows_by[h]=R
 print('h',h,'obs',len(x),'avgN',round(R.n.mean(),2),'coverage',round(R.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for nm,q in [('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('r360',slice('2028-11-29','2029-11-28')),('r180',slice('2029-05-29','2029-11-28'))]:
  y=R.loc[q,'ic']; print(nm,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
rows_by[10].to_csv('scripts/miner_3_20291129_drawdown_accel_ic.csv')
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20291129_drawdown_accel_signal.csv',index=False)
