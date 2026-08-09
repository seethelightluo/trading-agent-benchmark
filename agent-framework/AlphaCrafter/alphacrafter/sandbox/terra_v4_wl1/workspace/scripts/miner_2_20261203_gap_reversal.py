import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-02'); base='../persistent/stock_data'
cols={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()
 cols[s]=d[['open','close','high','low']].astype(float)
# aligned panel fields
P=pd.DataFrame({s:cols[s]['close'] for s in U}).sort_index(); P=P[P.index<=cut]
O=pd.DataFrame({s:cols[s]['open'] for s in U}).reindex(P.index)
H=pd.DataFrame({s:cols[s]['high'] for s in U}).reindex(P.index); L=pd.DataFrame({s:cols[s]['low'] for s in U}).reindex(P.index)
# gap reversal, scaled by recent true range to avoid price-level effects
prev=P.shift(1); gap=O/prev-1
tr=(H-L)/prev
f=(-gap/(tr.rolling(20,min_periods=10).median()+1e-8)).clip(-10,10)
# independent residualize current signal cross-sectionally against 5d reversal for audit
raw=-P.pct_change(5)
Y=P.shift(-1)/P-1
rows=[]
for dt in P.index:
 q=pd.concat([f.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(q)>=8: rows.append((dt,q.f.corr(q.y),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
print('dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
print('years',[(int(y),round(g.mean(),5),len(g)) for y,g in ic.groupby(ic.index.year)])
print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'corr_5d_reversal',round(f.stack().corr(raw.stack()),4))
for h in [5,10]:
 Y=P.shift(-h)/P-1; rr=[]
 for dt in P.index:
  q=pd.concat([f.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8: rr.append(q.f.corr(q.y))
 z=pd.Series(rr); print('H',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('cut',P.index.max())
