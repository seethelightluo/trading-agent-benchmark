import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv')
 if p.exists():
  z=pd.read_csv(p,parse_dates=['date']).sort_values('date'); D[s]=z.set_index('date')['close']
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# Candidate: short reversal scaled by own volatility, activated when cross-sectional dispersion is elevated.
rev=-px.pct_change(5); vol=r.rolling(20).std(); base=rev/(vol*np.sqrt(5)+1e-9)
disp=r.std(axis=1).rolling(20).mean(); threshold=disp.rolling(252,min_periods=100).median(); gate=(disp>threshold).astype(float)
f=base.mul(gate,axis=0)
rows=[]
for i in range(260,len(px)-10):
 a=f.iloc[i]; y=px.iloc[i+10]/px.iloc[i]-1; v=pd.concat([a,y],axis=1).dropna()
 if len(v)>=8 and v.iloc[:,0].std()>0: rows.append((px.index[i],v.iloc[:,0].corr(v.iloc[:,1]),len(v)))
q=pd.DataFrame(rows,columns=['date','ic','n']);
for lo,hi in [(None,None),(2020,2024),(2025,2026),(2027,2028),(2029,2029)]:
 w=q if lo is None else q[q.date.dt.year.between(lo,hi)]
 print('REGIME',lo,hi,'dates',len(w),'meanN',round(w.n.mean(),2),'IC',round(w.ic.mean(),6),'ICIR',round(w.ic.mean()/w.ic.std(ddof=1)*np.sqrt(252),4),'hit',round((w.ic>0).mean(),4))
print('coverage',q.n.mean()/15)
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20290920_dispersion_reversal_signal.csv',index=False); print('artifact',len(out),'dates',out.date.nunique(),'assets',out.symbol.nunique())
