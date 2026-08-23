import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-10-03')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for s in U}
px=pd.concat(D,axis=1).sort_index().ffill().loc[:END]; ret=px.pct_change(); csdisp=ret.rolling(20).std().mean(axis=1); med=csdisp.rolling(252,min_periods=126).median(); gate=(csdisp>med).astype(float)
base=-ret.rolling(5).sum()/ret.rolling(20).std(); candidates={'gated':base.mul(gate,axis=0),'inverted':-base.mul(gate,axis=0),'continuous':base.mul(csdisp/med,axis=0)}
for name,f in candidates.items():
 for h in [5,10,20]:
  ic=[]; dates=[]
  for i in range(len(px)-h):
   z=f.iloc[i]; nxt=px.iloc[i+h]/px.iloc[i]-1; ok=z.notna()&nxt.notna()
   if ok.sum()>=8:
    v=pd.Series(z[ok]).corr(pd.Series(nxt[ok]),method='spearman')
    if np.isfinite(v): ic.append(v); dates.append(px.index[i])
  a=np.array(ic); print(name,h,len(a),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),4),round(np.mean(a>0),4))
# save only provenance artifact for candidate inspection
candidates['continuous'].to_csv('scripts/miner_2_20291004_continuous_dispersion_signal.csv')
print('dates',len(px),'assets',len(px.columns),'range',px.index.min(),px.index.max())
