import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']) for s in U};P=pd.concat({s:d.set_index('date')['close'] for s,d in D.items()},axis=1).sort_index();R=np.log(P).diff()
# medium trend relative to peer median, normalized by trailing volatility; lagged
mom=R.rolling(20).sum(); v=R.rolling(40).std()*np.sqrt(252); f=mom.sub(mom.median(axis=1),axis=0)/(v+1e-8); f=f.clip(-8,8).shift(1); y=R.shift(-1)
ic=[]; dates=[]; ns=[]
for dt in f.index:
 ok=f.loc[dt].notna()&y.loc[dt].notna()
 if ok.sum()>=8:
  ic.append(f.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'));dates.append(dt);ns.append(ok.sum())
ic=pd.Series(ic,index=dates).dropna();print('dates',len(ic),'avg_n',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean());print('coverage',f.notna().mean().mean(),'turnover',((f.rank(axis=1,pct=True).diff().abs()).sum(axis=1)/f.notna().sum(axis=1)).mean())
for n in [3,5,10,20]:
 yy=R.shift(-1).rolling(n).sum().shift(-(n-1));z=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:z.append(f.loc[dt,ok].corr(yy.loc[dt,ok],method='spearman'))
 print('decay',n,np.nanmean(z))
for label,ix in [('early',ic.index<='2023-08-31'),('middle',(ic.index>'2023-08-31')&(ic.index<='2027-05-31')),('late',ic.index>'2027-05-31')]:print(label,ic[ix].mean(),ic[ix].count())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20301202_relative_trend20_vol40_signal.csv',index=False)
