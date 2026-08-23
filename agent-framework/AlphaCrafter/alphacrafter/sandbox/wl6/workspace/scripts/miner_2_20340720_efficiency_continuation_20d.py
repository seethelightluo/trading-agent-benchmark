import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 D[s]=x[['date','close']].drop_duplicates('date').set_index('date').sort_index()
P=pd.DataFrame({s:D[s].close.astype(float) for s in D}).sort_index().loc[:'2034-07-19'].ffill()
r=P.pct_change(); net=P/P.shift(20)-1; path=r.abs().rolling(20).sum()
eff=(net/path.replace(0,np.nan)).clip(-3,3); trend=P/P.shift(60)-1
F=(0.65*eff+0.35*trend).shift(1); F.to_csv('scripts/miner_2_20340720_efficiency_continuation_20d_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].rank().corr(z.iloc[:,1].rank());
   if pd.notna(v): vals.append(v); ns.append(len(z))
 a=np.asarray(vals); print(f'{h}D dates={len(a)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={np.mean(a):.8f} ICIR={np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)):.8f} hit={np.mean(a>0):.4f}')
rr=F.rank(axis=1,pct=True); print('assets',len(D),'date_range',P.index.min().date(),P.index.max().date(),'turnover_proxy',float(rr.diff().abs().mean(axis=1).dropna().mean()))
fr=P.shift(-20)/P-1
for label,lo,hi in [('2020-24',2020,2025),('2025-29',2025,2030),('2030-32',2030,2033),('2033-34',2033,2035)]:
 a=[]
 for dt in F.index:
  if not(lo<=dt.year<hi): continue
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('regime',label,'n',len(a),'IC',np.mean(a) if a else np.nan)
