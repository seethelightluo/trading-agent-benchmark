import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-12-20'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
# Candidate: breadth-confirmed trend efficiency. Risk-adjusted 20d trend, weighted by directional consistency and cross-sectional breadth.
vol=R.rolling(40,min_periods=25).std().shift(1)
trend=P.pct_change(20).shift(1).div(vol+1e-12)
cons=(R.shift(1).gt(0).rolling(20,min_periods=15).mean()-0.5)*2
breadth=R.shift(1).gt(0).mean(axis=1).rolling(10,min_periods=8).mean()
# retain trend when market breadth confirms; centered multiplier avoids pure market beta
F=trend*cons*(0.5+1.0*breadth.values[:,None])
F=pd.DataFrame(F,index=P.index,columns=P.columns)
F.to_csv('scripts/miner_3_20351221_breadth_confirmed_efficiency_signal.csv')
for h in [1,3,5,10]:
 vals=[]; cov=[]; dates=[]
 for i in range(45,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);cov.append(len(z)/15);dates.append(P.index[i])
 a=pd.Series(vals,index=dates)
 print('h',h,'dates',len(a),'avgN',round(np.mean(cov)*15,2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'recent252',round(a.tail(252).mean(),6),'recentIR',round(a.tail(252).mean()/a.tail(252).std(ddof=1),6))
 print('blocks',*[round(x.mean(),6) for x in np.array_split(a,4)])
print('turn',round(F.rank(axis=1).diff().abs().mean().mean()/15,6),'coverage',round(F.notna().mean().mean(),4))
