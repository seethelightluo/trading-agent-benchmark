import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}
P=pd.DataFrame(D).sort_index().loc[:'2034-10-24'].ffill(); r=P.pct_change(); v20=r.rolling(20,min_periods=15).std(); v5=r.rolling(5,min_periods=5).std()
tr=(v5.mean(axis=1)/v20.mean(axis=1)).clip(.5,2.5)
# Contrarian standardized 5D shock, activated during moderately elevated volatility transition; one-day lag.
F=(-(P/P.shift(5)-1)/(v20*np.sqrt(5)).replace(0,np.nan)).mul((tr>1.05).astype(float),axis=0).shift(1)
F.to_csv('scripts/miner_2_20341026_smooth_transition_reversal_10d_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fw=P.shift(-h)/P-1; ics=[]; ns=[]; dates=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
   if pd.notna(q): ics.append(q); ns.append(len(z)); dates.append(dt)
 a=np.asarray(ics); mid=np.array([d.year<2030 for d in dates])
 print(f'{h}D dates={len(a)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={np.mean(a):.8f} ICIR={np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)):.8f} hit={np.mean(a>0):.4f}')
 if h==10:
  for label,m in [('2020-24',[d.year<=2024 for d in dates]),('2025-29',[2025<=d.year<=2029 for d in dates]),('2030-32',[2030<=d.year<=2032 for d in dates]),('2033-34',[d.year>=2033 for d in dates])]:
   aa=a[np.asarray(m)]; print(label,'n',len(aa),'IC',round(float(np.mean(aa)),8) if len(aa) else None)
print('gate_fraction',float((tr>1.05).mean()))
