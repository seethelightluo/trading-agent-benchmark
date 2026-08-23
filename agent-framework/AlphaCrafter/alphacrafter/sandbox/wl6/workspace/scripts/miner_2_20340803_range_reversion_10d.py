import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 D[s]=x[['date','close']].drop_duplicates('date').set_index('date').sort_index()
P=pd.DataFrame({s:D[s].close.astype(float) for s in D}).sort_index().loc[:'2034-08-02'].ffill()
r=P.pct_change()
# Contrarian signal: recent loss relative to its trailing range, scaled by realized volatility.
ret10=P/P.shift(10)-1
hi=P.rolling(60,min_periods=40).max(); lo=P.rolling(60,min_periods=40).min()
loc=((P-lo)/(hi-lo).replace(0,np.nan)-0.5)
vol=r.rolling(20,min_periods=15).std()
F=(-0.70*ret10/(vol*np.sqrt(10)).replace(0,np.nan)-0.30*loc).shift(1)
F.to_csv('scripts/miner_2_20340803_range_reversion_10d_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
   if pd.notna(v): vals.append(v); ns.append(len(z))
 a=np.asarray(vals)
 print(f'{h}D dates={len(a)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={np.mean(a):.8f} ICIR={np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)):.8f} hit={np.mean(a>0):.4f}')
rr=F.rank(axis=1,pct=True); print('assets',len(D),'date_range',P.index.min().date(),P.index.max().date(),'turnover_proxy',float(rr.diff().abs().mean(axis=1).dropna().mean()))
fr=P.shift(-10)/P-1
for label,loy,hiy in [('2020-24',2020,2025),('2025-29',2025,2030),('2030-32',2030,2033),('2033-34',2033,2035)]:
 a=[]
 for dt in F.index:
  if not(loy<=dt.year<hiy): continue
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('regime',label,'n',len(a),'IC',np.mean(a) if a else np.nan)
