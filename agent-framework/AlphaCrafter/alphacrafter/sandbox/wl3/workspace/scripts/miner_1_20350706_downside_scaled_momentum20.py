import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); cutoff=pd.Timestamp('2035-07-04'); c=c[c.index<=cutoff]
r=c.pct_change()
# Sortino-style medium-term momentum: cumulative 20-session return scaled by downside deviation over the same window.
downside=r.where(r<0,0.0).rolling(20,min_periods=15).std()
factor=c.pct_change(20)/(downside+1e-12)
# lag one completed session to avoid using the decision day's close
factor=factor.shift(1)
def calc(h):
 rows=[]
 for dt in factor.index:
  y=c.pct_change(h).shift(-h).loc[dt]; x=factor.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=z.ic.dropna(); return z,a
z,a=calc(10)
print('factor=downside_scaled_momentum20'); print('dates',len(z),'instruments',15,'avg_n',z.n.mean(),'coverage',z.n.mean()/15)
print('ic %.8f icir %.8f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),(a>0).mean()))
for h in [1,5,10,20]:
 q=[]
 for dt in factor.index:
  y=c.pct_change(h).shift(-h).loc[dt]; x=factor.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('horizon',h,'ic',np.nanmean(q),'n',len(q))
for name,x in [('early',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:]),('recent120',a.tail(120))]: print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
rank=factor.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).dropna().mean(),'period',z.index.min().date(),z.index.max().date())
sig=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); sig.to_csv('scripts/miner_1_20350706_downside_scaled_momentum20_signal.csv',index=False)
