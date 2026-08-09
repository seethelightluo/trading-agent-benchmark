import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); D[s]=d[d.index<=end]
# No forward fill: only genuine same-day OHLC bars; forward return requires same asset future observation.
def run(kind):
 C=pd.DataFrame({s:d.close for s,d in D.items()}); O=pd.DataFrame({s:d.open for s,d in D.items()}); H=pd.DataFrame({s:d.high for s,d in D.items()}); L=pd.DataFrame({s:d.low for s,d in D.items()})
 rng=(H-L).replace(0,np.nan); clv=2*(C-L)/rng-1; candle=(C-O)/rng
 base=-(.6*clv+.4*candle)
 F=base.rolling(3,min_periods=3).mean() if kind=='smooth3' else base
 # no-fill forward return aligned by asset, naturally shift(-h) row calendar is wrong; use each asset's next h valid observations
 out={}
 for h in [1,5,10,20]:
  vals=[]; ns=[]; dates=[]
  for dt in F.index:
   xs=[]; ys=[]
   for s in U:
    if dt not in D[s].index or pd.isna(F.loc[dt,s]): continue
    ix=D[s].index.get_loc(dt); j=ix+h
    if j<len(D[s]) and pd.notna(D[s].iloc[j].close) and D[s].iloc[j].close!=0:
     xs.append(F.loc[dt,s]); ys.append(D[s].iloc[j].close/D[s].iloc[ix].close-1)
   if len(xs)>=8 and len(set(xs))>1: vals.append(pd.Series(xs).corr(pd.Series(ys),method='spearman')); ns.append(len(xs)); dates.append(dt)
  a=np.array(vals); out[h]=(len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
 # turnover on genuine daily ranks, no-fill rows
 ranks=F.rank(axis=1,pct=True); turn=(ranks-ranks.shift()).abs().mean(axis=1).dropna()
 print(kind,'metrics',out,'coverage_dates',sum(F.notna().sum(axis=1)>=8)/len(F),'turn',turn.mean())
run('raw');run('smooth3')
# correlation exact pooled valid observations against library-like definitions, no fill
C=pd.DataFrame({s:d.close for s,d in D.items()}); F=-(.6*(2*(pd.DataFrame({s:d.close for s,d in D.items()})-pd.DataFrame({s:d.low for s,d in D.items()}))/((pd.DataFrame({s:d.high for s,d in D.items()})-pd.DataFrame({s:d.low for s,d in D.items()})).replace(0,np.nan))-1)+.4*(pd.DataFrame({s:d.close for s,d in D.items()})-pd.DataFrame({s:d.open for s,d in D.items()}))/((pd.DataFrame({s:d.high for s,d in D.items()})-pd.DataFrame({s:d.low for s,d in D.items()})).replace(0,np.nan))).rolling(3,min_periods=3).mean()
for nm,X in [('mom20',C/C.shift(20)-1),('rev5',-(C/C.shift(5)-1)),('ram20',(C/C.shift(20)-1)/C.pct_change().rolling(60).std())]:
 z=pd.concat([F.stack(),X.stack()],axis=1).dropna();print('corr',nm,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))
