import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-07-11']; r=P.pct_change()
# Candidate: volatility-spread rotation. Prefer assets whose short volatility is low relative to
# their medium volatility, while retaining medium-horizon relative strength.
v5=r.rolling(5,min_periods=5).std(); v30=r.rolling(30,min_periods=20).std()
relvol=(v5/v30).replace([np.inf,-np.inf],np.nan)
mom=r.rolling(10,min_periods=10).sum()
factor=(mom/(v30*np.sqrt(10))).shift(1) * (1-relvol.shift(1)).clip(-1,1)
rows=[]
for d in P.index:
 z=pd.concat([factor.loc[d],(P.shift(-5)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for n,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-07-01','2029-07-11')),('recent180',slice('2029-01-01','2029-07-11'))]:
 x=R.loc[q,'ic']; print(n,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for h in [1,5,10,20]:
 x=[]; fw=P.shift(-h)/P-1
 for d in P.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(x);print('decay',h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20290712_volspread_rotation_signal.csv',index=False)
# Revalidation of prior effective factor
asym=(r.clip(upper=0).rolling(20,min_periods=15).std()/r.rolling(20,min_periods=15).std()).replace([np.inf,-np.inf],np.nan).clip(.25,2).shift(1)
old=(-r.rolling(5,min_periods=5).sum().shift(1))*asym
rows=[]
for d in P.index:
 z=pd.concat([old.loc[d],(P.shift(-5)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
x=pd.Series(rows);print('OLD_REVALIDATION',len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
