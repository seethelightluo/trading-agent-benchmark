import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c[c.index<=pd.Timestamp('2035-06-21')]; r=c.pct_change()
up=r.clip(lower=0).rolling(20,min_periods=15).sum(); dn=(-r.clip(upper=0)).rolling(20,min_periods=15).sum(); factor=(up-dn)/(up+dn+1e-12)
rows=[]
for dt in factor.index:
 y=c.pct_change(10).shift(-10).loc[dt]; x=factor.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=z.ic.dropna()
def stats(x): return len(x),float(x.mean()),float(x.std(ddof=1)),float(x.mean()/(x.std(ddof=1)+1e-12)),float((x>0).mean())
print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.mean()/15);print('stats',stats(a),'recent120',stats(a.tail(120)),'recent252',stats(a.tail(252)))
print('blocks',[stats(a.iloc[i*len(a)//4:(i+1)*len(a)//4]) for i in range(4)])
for h in [1,5,10,20]:
 q=[]
 for dt in factor.index:
  y=c.pct_change(h).shift(-h).loc[dt]; x=factor.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,stats(pd.Series(q).dropna()))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
sig=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();sig.to_csv('scripts/miner_2_20350622_return_asymmetry20_signal.csv',index=False)
