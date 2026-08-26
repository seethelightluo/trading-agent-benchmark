import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']) for s in U}
P=pd.concat({s:d.set_index('date')['close'] for s,d in D.items()},axis=1).sort_index(); R=np.log(P).diff()
r5=R.rolling(5).sum(); r20=R.rolling(20).sum(); vol=R.rolling(40).std()*np.sqrt(252)
# Acceleration: long trend net of recent 5d move, normalized by risk, lagged to avoid lookahead.
f=((r20-r5)/(vol+1e-8)).clip(-8,8).shift(1)
y=R.shift(-1).rolling(10).sum().shift(-9)
vals=[]; dates=[]; ns=[]
for dt in f.index:
 ok=f.loc[dt].notna()&y.loc[dt].notna()
 if ok.sum()>=8:
  z=f.loc[dt,ok].corr(y.loc[dt,ok],method='spearman')
  if pd.notna(z): vals.append(z);dates.append(dt);ns.append(ok.sum())
s=pd.Series(vals,index=dates)
print('dates',len(s),'avg_n',round(np.mean(ns),2),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
for n in [1,3,5,10,20]:
 yy=R.shift(-1).rolling(n).sum().shift(-(n-1));z=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:z0=f.loc[dt,ok].corr(yy.loc[dt,ok],method='spearman'); z += [z0] if pd.notna(z0) else []
 print('decay',n,np.nanmean(z))
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(axis=1,pct=True).diff().abs()).sum(axis=1)/f.notna().sum(axis=1)).mean())
for label,ix in [('early',s.index<='2023-08-31'),('middle',(s.index>'2023-08-31')&(s.index<='2027-05-31')),('late',s.index>'2027-05-31')]:print(label,s[ix].mean(),ix.sum())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20301230_acceleration_signal.csv',index=False)
