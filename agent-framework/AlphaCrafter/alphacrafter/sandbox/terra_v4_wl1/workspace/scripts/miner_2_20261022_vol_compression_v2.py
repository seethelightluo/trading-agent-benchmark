import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data';cut=pd.Timestamp('2026-10-22')
F={};P={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].astype(float);d=d[d.index<=cut];r=d.pct_change();P[s]=d
 v10=r.rolling(10,min_periods=8).std();v60=r.rolling(60,min_periods=40).std();F[s]=-(v10/v60)
F=pd.DataFrame(F); P=pd.DataFrame(P)
for h in [1,3,5,10]:
 rows=[]
 for dt in F.index:
  vals=[]
  for s in U:
   # forward by asset's next h observations
   ser=P[s].dropna();
   if dt not in ser.index: continue
   i=ser.index.get_loc(dt)
   if i+h>=len(ser): continue
   y=ser.iloc[i+h]/ser.iloc[i]-1; x=F.at[dt,s]
   if pd.notna(x): vals.append((x,y))
  if len(vals)>=8: rows.append((dt,pd.Series([x for x,y in vals]).corr(pd.Series([y for x,y in vals])),len(vals)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');ic=a.ic
 print('H',h,'dates',len(ic),'avgN',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YEAR',yr,round(g.mean(),5),len(g))
rk=F.rank(axis=1,pct=True);print('coverage',F.notna().sum().sum()/F.size,'turnover',rk.diff().abs().mean().mean())
