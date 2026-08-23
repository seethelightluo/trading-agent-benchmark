import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2030-01-08']
r=P.pct_change()
# Candidate: liquidity/volatility-normalized short-term reversal. Lagged 5d return
# divided by lagged 20d volatility, with cross-sectional clipping for stability.
f=-(P.shift(1)/P.shift(6)-1)/(r.rolling(20,min_periods=15).std().shift(1)*np.sqrt(5))
f=f.clip(-5,5)
fw=P.shift(-10)/P-1
rows=[]
for d in P.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
print('obs',len(x),'start',R.index.min(),'end',R.index.max(),'avgN',round(R.n.mean(),2),'coverage',round(R.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for nm,q in [('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('2029',slice('2029','2029')),('r360',slice('2029-01-08','2030-01-08')),('r180',slice('2029-07-08','2030-01-08'))]:
 y=R.loc[q,'ic']; print(nm,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20300110_volnorm_reversal_signal.csv',index=False)
R.to_csv('scripts/miner_1_20300110_volnorm_reversal_ic.csv')
