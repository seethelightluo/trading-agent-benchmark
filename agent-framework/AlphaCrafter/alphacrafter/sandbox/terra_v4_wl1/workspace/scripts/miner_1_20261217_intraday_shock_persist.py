import numpy as np,pandas as pd,glob,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]; O=pd.DataFrame({s:D[s].open for s in U}).reindex(P.index); H=pd.DataFrame({s:D[s].high for s in U}).reindex(P.index); L=pd.DataFrame({s:D[s].low for s in U}).reindex(P.index)
tr=(H-L).combine((H-P.shift(1)).abs(),np.maximum).combine((L-P.shift(1)).abs(),np.maximum); atr=tr.shift(1).rolling(20,min_periods=15).mean(); shock=-(P-O)/atr; clv=((P-L)-(H-P))/(H-L).replace(0,np.nan); f=(.7*shock+.3*(-clv)).replace([np.inf,-np.inf],np.nan)
rows=[]; y=P.shift(-1).div(P)-1
for d in P.index:
 q=pd.concat([f.loc[d].rename('signal'),y.loc[d].rename('fwd')],axis=1).dropna()
 if len(q)>=8: rows += [(d,s,v) for s,v in f.loc[d].dropna().items()]
out=pd.DataFrame(rows,columns=['date','symbol','signal']); out.to_csv('scripts/miner_1_20261217_intraday_shock_signal.csv',index=False)
ics=[]
for d in P.index:
 q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8: ics.append(q.iloc[:,0].corr(q.iloc[:,1]))
ic=pd.Series(ics); print(len(ic),ic.mean(),ic.mean()/ic.std(ddof=1),f.notna().mean().mean())
# artifact correlation audit against available CSVs, aligned signal values
wide=out.pivot(index='date',columns='symbol',values='signal').stack()
mx=0; pair=''
for fn in glob.glob('scripts/*signal.csv'):
 try:
  z=pd.read_csv(fn,parse_dates=['date']); cols=[c for c in ['signal','factor','value','score'] if c in z]
  if not cols or not {'date','symbol'}.issubset(z): continue
  a=z.set_index(['date','symbol'])[cols[0]].astype(float); q=pd.concat([wide,a],axis=1).dropna()
  if len(q)>20:
   r=abs(q.iloc[:,0].corr(q.iloc[:,1]));
   if fn.endswith('intraday_shock_signal.csv'): continue
   if r>mx: mx=r;pair=fn
 except Exception: pass
print('MAXCORR',mx,pair)
