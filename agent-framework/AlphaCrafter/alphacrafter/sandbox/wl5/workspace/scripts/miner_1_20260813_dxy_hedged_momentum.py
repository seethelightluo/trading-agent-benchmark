import pandas as pd, numpy as np, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ser(s):
 p=('../persistent/index_data/' if s=='DXY' else '../persistent/stock_data/')+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); return d.close.astype(float)
px=pd.DataFrame({s:ser(s) for s in U}).join(ser('DXY').rename('DXY'),how='inner').sort_index()
r=px[U].pct_change(); dr=px.DXY.pct_change()
beta=r.rolling(60,min_periods=45).cov(dr).div(dr.rolling(60,min_periods=45).var(),axis=0)
f=r.rolling(20,min_periods=15).sum()-beta.mul(dr.rolling(20,min_periods=15).sum(),axis=0)
for h in [1,5,10]:
 vals=[]
 for i in range(len(px)-h):
  x=f.iloc[i]; y=(px[U].iloc[i+h]/px[U].iloc[i]-1); z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(vals); print('H',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
valid=f.notna().sum(axis=1); print('coverage',valid.mean()/15,'mean names',valid.mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for yr in range(2020,2027):
 a=[]
 for i in range(len(px)-1):
  if px.index[i].year!=yr: continue
  z=pd.concat([f.iloc[i],(px[U].iloc[i+1]/px[U].iloc[i]-1)],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 if a: print('REG',yr,len(a),round(np.mean(a),5),round(np.mean(a)/np.std(a,ddof=1),5))
print('period',px.index.min(),px.index.max(),'assets',len(U))
