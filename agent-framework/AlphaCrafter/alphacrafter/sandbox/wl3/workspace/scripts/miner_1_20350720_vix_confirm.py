import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c[c.index<=pd.Timestamp('2035-07-19')]; r=c.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.astype(float).reindex(c.index).ffill()
z=(v-v.rolling(120,min_periods=60).mean())/(v.rolling(120,min_periods=60).std()+1e-12)
# Candidate: confirmed 20d momentum scaled by 60d total volatility, with milder VIX stress attenuation
stress=(1-0.20*z).clip(0.55,1.25)
f=c.pct_change(20)/(r.rolling(60,min_periods=40).std()+1e-12)*np.sign(c.pct_change(60))*stress.values[:,None]
rows=[]
for h in [1,5,10,20]:
 y=c.pct_change(h).shift(-h); q=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:q.append((dt,f.loc[dt][ok].rank().corr(y.loc[dt][ok].rank()),ok.sum()))
 a=pd.DataFrame(q,columns=['date','ic','n']).set_index('date'); x=a.ic.dropna()
 print('horizon',h,'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15,'IC',x.mean(),'ICIR',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
 if h==10:
  for name,zx in [('early',x.iloc[:len(x)//3]),('middle',x.iloc[len(x)//3:2*len(x)//3]),('recent',x.iloc[2*len(x)//3:]),('recent120',x.tail(120))]:print(name,len(zx),zx.mean(),zx.mean()/(zx.std(ddof=1)+1e-12)*np.sqrt(len(zx)))
  print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date())
  f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350720_vix_confirm_signal.csv',index=False)
