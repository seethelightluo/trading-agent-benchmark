import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float).replace(0,np.nan)
 except: pass
p=pd.DataFrame(D).sort_index(); lr=np.log(p).diff()
# Novel candidate: medium reversal scaled by the change in downside risk.
down20=lr.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
down60=lr.clip(upper=0).pow(2).rolling(60,min_periods=40).mean().pow(.5)
r40=np.log(p/p.shift(40))
raw=-r40/(down60+1e-10)
# favor reversals whose recent downside risk is compressing, while dampening expanding-risk names
risk=np.log((down20+1e-10)/(down60+1e-10))
f=(raw*(1-risk.clip(-1,1))).rolling(3,min_periods=2).mean()
rows=[]; sig=[]
for d in f.index:
 q=np.log(p.shift(-10)/p); z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): rows.append((d,c,len(z)))
  for s in z.index:sig.append((d,s,float(f.loc[d,s])))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]).sort_index(); ns=np.array([x[2] for x in rows])
print('dates',len(i),'avgN',ns.mean(),'coverage',ns.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for k in [365,750,1260]:
 z=i.tail(k); print('recent',k,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for a,b in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 z=i.loc[a:b];print('regime',a+'-'+b,len(z),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 q=np.log(p.shift(-h)/p); a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(a))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20330217_downside_risk_transition_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i.values,'n':ns}).to_csv('scripts/miner_2_20330217_downside_risk_transition_ic.csv',index=False)
