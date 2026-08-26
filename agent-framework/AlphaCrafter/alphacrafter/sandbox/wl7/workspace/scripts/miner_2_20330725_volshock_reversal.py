import pandas as pd,numpy as np
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-07-24'); P={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]
 P[a]=d.close
P=pd.DataFrame(P); r=P.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
shock=(v20/v60).clip(lower=0.5,upper=3.0)
# Contrarian 5-session return, risk-normalized and amplified only when recent volatility exceeds its 60d baseline; lag one day.
fac=((-P.pct_change(5)/(v20*np.sqrt(5)))*shock).shift(1)
print('dates',len(fac),'assets',len(assets),'overall_coverage %.4f'%fac.notna().mean().mean())
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[];ns=[]; ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(len(z));ds.append(dt)
 x=np.array(vals); print('H',h,'dates',len(x),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f thirds'% (x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)),[round(q.mean(),6) for q in np.array_split(x,3)])
 # recent 504 valid IC observations
 print(' recent504 IC %.6f ICIR %.6f hit %.4f n %d'%(x[-504:].mean(),x[-504:].mean()/x[-504:].std(ddof=1),np.mean(x[-504:]>0),min(504,len(x))))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
out=[{'date':dt.date(),'asset':a,'signal':fac.loc[dt,a]} for dt in fac.index for a in assets]
pd.DataFrame(out).to_csv('scripts/miner_2_20330725_volshock_reversal_signal.csv',index=False)
