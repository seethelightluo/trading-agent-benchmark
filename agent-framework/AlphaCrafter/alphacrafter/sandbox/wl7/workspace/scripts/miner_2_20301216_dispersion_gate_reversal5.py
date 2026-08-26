import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']) for s in U}
P=pd.concat({s:d.set_index('date')['close'] for s,d in D.items()},axis=1).sort_index(); R=np.log(P).diff()
raw=R.rolling(5).sum(); vol=R.rolling(20).std()*np.sqrt(252)
rel=raw.sub(raw.median(axis=1),axis=0); base=(-rel/(vol+1e-8)).clip(-8,8)
disp=R.sub(R.median(axis=1),axis=0).abs().median(axis=1)
gate=disp>disp.rolling(120,min_periods=60).quantile(.70)
f=base.where(gate, np.nan).shift(1); fwd=R.shift(-1)
rows=[]; dates=[]; ns=[]
for dt in f.index:
 x=f.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append(x[ok].corr(y[ok],method='spearman'));dates.append(dt);ns.append(ok.sum())
ic=pd.Series(rows,index=pd.to_datetime(dates)).dropna()
print('dates',len(ic),'avg_n',np.mean(ns),'active_frac',gate.mean());print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for n in [3,5,10,20]:
 yy=R.shift(-1).rolling(n).sum().shift(-(n-1));z=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:z.append(f.loc[dt,ok].corr(yy.loc[dt,ok],method='spearman'))
 print('decay',n,np.nanmean(z))
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(axis=1,pct=True).diff().abs()).sum(axis=1)/f.notna().sum(axis=1)).replace([np.inf,-np.inf],np.nan).mean())
for label,ix in [('middle',(ic.index>'2023-08-31')&(ic.index<='2027-05-31')),('late',ic.index>'2027-05-31')]:print(label,ic[ix].mean(),ic[ix].count())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20301216_dispersion_gate_reversal5_signal.csv',index=False)
