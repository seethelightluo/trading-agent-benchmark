import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2030-01-23']
r=P.pct_change()
# Momentum conditioned on volatility state: recent 20d trend, scaled by 20d risk and
# rewarded when recent risk is below its 60d baseline (all inputs lagged one day).
ret20=P.shift(1)/P.shift(21)-1
v20=r.rolling(20,min_periods=15).std().shift(1)
v60=r.rolling(60,min_periods=40).std().shift(1)
state=(v60-v20)/v60.replace(0,np.nan)
f=ret20/v20.replace(0,np.nan)*(1+0.5*state.clip(-1,1))

def calc(h):
 fw=P.shift(-h)/P-1; rows=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
 print('H',h,'obs',len(x),'range',R.index.min().date(),R.index.max().date(),'avgN',round(R.n.mean(),2),'coverage',round(R.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for nm,q in [('2020-23',slice('2020','2023')),('2024-26',slice('2024','2026-07-15')),('online',slice('2026-07-16','2028-12-31')),('2029',slice('2029','2029')),('r360',slice('2029-01-24','2030-01-23')),('r180',slice('2029-07-28','2030-01-23'))]:
  y=R.loc[q,'ic']; print(nm,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)
 return R
R=calc(10)
for h in [5,20]: calc(h)
turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()
print('turnover',round(turn,6),'signal rows',int(f.notna().sum().sum()))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20300124_volatility_state_momentum_signal.csv',index=False)
R.to_csv('scripts/miner_2_20300124_volatility_state_momentum_ic.csv')
