import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c.loc[c.index<=pd.Timestamp('2035-06-20')]; r=c.pct_change()
# Medium-horizon trend normalized by realized volatility, rewarding persistent returns per unit of risk.
sig=c.pct_change(30)/(r.rolling(20,min_periods=15).std()*np.sqrt(30)+1e-12)
rows=[]
for dt in sig.index:
 f=c.pct_change(10).shift(-10).loc[dt]; x=sig.loc[dt]; ok=x.notna()&f.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],f[ok]).statistic,ok.sum()))
r0=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=r0.ic
print('factor=quiet_trend30'); print('dates',len(a),'avg_n',r0.n.mean(),'coverage',r0.n.mean()/15,'period',r0.index.min().date(),r0.index.max().date())
print('h10_ic %.8f icir %.8f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),(a>0).mean()))
for h in [1,5,10,20]:
 z=[]
 f=c.pct_change(h).shift(-h)
 for dt in sig.index:
  x=sig.loc[dt]; y=f.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic)
 print('horizon',h,'ic %.8f n %d'%(np.nanmean(z),len(z)))
for name,x in [('early',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:]),('recent120',a.tail(120))]: print(name,len(x),'ic %.8f icir %.5f hit %.4f'%(x.mean(),x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),(x>0).mean()))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350622_quiet_trend30_signal.csv',index=False)
