import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'] for a in assets},axis=1).sort_index().loc[:'2029-05-16']
r=P.pct_change(); down=r.clip(upper=0).rolling(20,min_periods=15).std(); tot=r.rolling(20,min_periods=15).std()
asym=(down/tot).replace([np.inf,-np.inf],np.nan).clip(.25,2.0)
base=(-r.rolling(5,min_periods=5).sum().shift(1)); raw=base*asym.shift(1)
# Cross-sectional residual: remove linear exposure to the ordinary 5d reversal each date.
def resid(row, xrow):
 z=pd.concat([row.rename('y'),xrow.rename('x')],axis=1).dropna()
 if len(z)<8: return row* np.nan
 xc=z.x-z.x.mean(); yc=z.y-z.y.mean(); den=(xc*xc).sum()
 beta=(xc*yc).sum()/den if den>1e-12 else 0
 out=z.y-beta*z.x
 return out.reindex(row.index)
factor=pd.DataFrame([resid(raw.loc[d],base.loc[d]) for d in P.index],index=P.index,columns=P.columns)
rows=[]
for d in P.index:
 z=pd.concat([factor.loc[d],(P.shift(-5)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for name,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-05-16','2029-05-16')),('recent180',slice('2028-11-16','2029-05-16'))]:
 x=R.loc[q,'ic']; print(name,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 vals=[]; fw=P.shift(-h)/P-1
 for d in P.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
rank=factor.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
# correlations with raw and prior artifact if available
for name,s in [('raw',raw)]:
 a=factor.stack(); b=s.stack(); q=pd.concat([a,b],axis=1).dropna(); print('corr_'+name, q.iloc[:,0].corr(q.iloc[:,1]))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20290517_residual_downside_reversal_signal.csv',index=False)
