import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2030-02-06'
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:cut]
r=P.pct_change(); base=-(P.shift(1)/P.rolling(120,min_periods=80).max().shift(1)-1)/r.rolling(20,min_periods=15).std().shift(1); tr=P.shift(1)/P.shift(61)-1
f=base*(1+10*tr.clip(lower=0)); fw=P.shift(-10)/P-1; out=[]
for d in P.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna();
 if len(z)>=8:
  c=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(c):out.append((d,c,len(z)))
R=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); x=R.ic
print('obs',len(x),'range',R.index.min().date(),R.index.max().date(),'avgN',round(R.n.mean(),2),'coverage',round(R.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for nm,q in [('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('2029',slice('2029','2029')),('r360',slice('2029-02-06','2030-02-06')),('r180',slice('2029-08-11','2030-02-06'))]:
 y=R.loc[q,'ic'];print(nm,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'rows',int(f.notna().sum().sum()))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20300207_recovery_boost_signal.csv',index=False);R.to_csv('scripts/miner_3_20300207_recovery_boost_ic.csv')
