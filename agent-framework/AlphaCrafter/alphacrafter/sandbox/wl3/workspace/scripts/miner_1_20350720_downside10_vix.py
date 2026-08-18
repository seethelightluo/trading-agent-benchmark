import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'));d.date=pd.to_datetime(d.date);p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index();c=c[c.index<=pd.Timestamp('2035-07-19')];r=c.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.astype(float).reindex(c.index).ffill();z=(v-v.rolling(120,min_periods=60).mean())/(v.rolling(120,min_periods=60).std()+1e-12); stress=(1-.20*z).clip(.55,1.25)
# Shorter momentum over downside deviation, conditioned by macro stress.
down=r.where(r<0).rolling(20,min_periods=12).std();f=c.pct_change(10)/(down+1e-12)*stress.values[:,None]
for h in [1,5,10,20]:
 y=c.pct_change(h).shift(-h);rows=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:rows.append((dt,f.loc[dt][ok].rank().corr(y.loc[dt][ok].rank()),ok.sum()))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');x=a.ic.dropna();print('horizon',h,'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15,'IC',x.mean(),'ICIR',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
 if h==10:
  for nm,q in [('early',x.iloc[:len(x)//3]),('middle',x.iloc[len(x)//3:2*len(x)//3]),('recent',x.iloc[2*len(x)//3:]),('recent120',x.tail(120))]:print(nm,len(q),q.mean(),q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)))
  print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date());f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350720_downside10_vix_signal.csv',index=False)
