import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-09-05']
r=P.pct_change(); b=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(b).div(b.rolling(60,min_periods=40).var(),axis=0)
res=r.rolling(5,min_periods=5).sum().sub(beta.mul(b.rolling(5,min_periods=5).sum(),axis=0))
vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# fade only negative residual shocks; neutral/positive observations are excluded
raw=(-res.div(vol.replace(0,np.nan))).shift(1)
factor=raw.where(raw>0)
rows=[]
for d in P.index:
 z=pd.concat([factor.loc[d],(P.shift(-5)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for nm,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-09-05','2029-09-05')),('recent180',slice('2029-03-01','2029-09-05'))]:
 x=R.loc[q,'ic'];print(nm,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for h in [1,5,10,20]:
 vals=[];fw=P.shift(-h)/P-1
 for d in P.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals);print('decay',h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20290906_downside_residual_reversal_signal.csv',index=False)
