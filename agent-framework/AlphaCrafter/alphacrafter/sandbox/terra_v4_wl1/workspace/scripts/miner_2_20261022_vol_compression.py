import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-10-22')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].astype(float); px[s]=d[d.index<=cut]
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# volatility compression: low recent realized vol relative to medium vol, with sign from medium momentum
v10=R.rolling(10,min_periods=8).std(); v60=R.rolling(60,min_periods=40).std(); f=-(v10/v60) # low vol favored
# evaluate forward close-to-close returns, date aligned
for h in [1,3,5,10]:
 out=[]
 for dt in R.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.DataFrame({'f':f.loc[dt],'y':y}).dropna()
  if len(q)>=8: out.append((dt,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YEAR',yr,round(g.mean(),5),len(g))
rk=f.rank(axis=1,pct=True); print('coverage',f.notna().sum().sum()/f.size,'turnover',rk.diff().abs().mean().mean())
print('corr raw vol60',f.stack().corr((-v60).stack()))
