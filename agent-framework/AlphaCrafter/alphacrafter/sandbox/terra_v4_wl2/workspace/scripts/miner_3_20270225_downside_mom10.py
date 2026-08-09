import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if not os.path.exists(p): print('missing',s); continue
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# continuous downside-risk-adjusted medium momentum, lagged naturally via signal at t and forward t+1..5
mom=P.pct_change(10); down=R.clip(upper=0).rolling(30).std(); fac=mom/(down*np.sqrt(30)+1e-8)
fwd=P.shift(-5)/P-1
rows=[]; sig=[]
for dt in fac.index:
 x=fac.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z))); sig.append((dt,*[fac.loc[dt,s] for s in U]))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(a),'avgN',a.n.mean(),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(),'hit',(a.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 q=a.loc[lo:hi]; print(lo, len(q), q.ic.mean() if len(q) else None, q.ic.mean()/q.ic.std() if len(q)>1 else None)
# turnover rank changes
S=pd.DataFrame(sig,columns=['date']+U).set_index('date').rank(axis=1,pct=True); print('turn', (S.diff().abs().mean(axis=1).mean()))
out=pd.DataFrame(sig,columns=['date']+U); out.to_csv('../persistent/factor_signals_miner_3_20270225_downside_mom10.csv',index=False)
# decay
for h in [1,5,10]:
 fw=P.shift(-h)/P-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,len(rr),np.mean(rr),np.mean(rr)/np.std(rr))
