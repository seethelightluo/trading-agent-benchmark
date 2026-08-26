import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2034-05-09']
r=p.pct_change(20); vol=p.pct_change().rolling(40).std()*np.sqrt(20); z=r/vol.replace(0,np.nan)
med=r.median(axis=1); disp=r.sub(med,axis=0).abs().median(axis=1)
dm=disp.rolling(120,min_periods=60).mean(); ds=disp.rolling(120,min_periods=60).std(); high=disp>dm+0.25*ds
f=z.mul(np.where(high,1.0,-1.0),axis=0); fr=p.shift(-10).div(p)-1
rows=[]; dates=[]; ns=[]
for dt in p.index:
 m=f.loc[dt].notna()&fr.loc[dt].notna()
 if m.sum()>=8: rows.append(spearmanr(f.loc[dt,m],fr.loc[dt,m]).statistic); dates.append(dt); ns.append(m.sum())
ic=pd.Series(rows,index=dates)
print('dates',len(ic),'range',dates[0],dates[-1],'meanN',np.mean(ns),'n_assets',len(U)); print('IC',ic.mean(),'ICIR_ann',ic.mean()/ic.std(ddof=1)*np.sqrt(252),'hit',(ic>0).mean(),'std',ic.std())
for h in [5,10,20]:
 yy=p.shift(-h).div(p)-1;a=[]
 for dt in p.index:
  m=f.loc[dt].notna()&yy.loc[dt].notna()
  if m.sum()>=8:a.append(spearmanr(f.loc[dt,m],yy.loc[dt,m]).statistic)
 print('decay',h,np.nanmean(a))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',f.notna().mean().mean())
for a,b in [('2025','2027'),('2028','2029'),('2030','2032'),('2033','2034')]:
 q=ic.loc[a:b];print('regime',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
f.stack().rename('signal').to_csv('scripts/miner_1_20340511_dispersion_conditioned_momentum_signal.csv',header=True)
