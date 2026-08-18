import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def s(x):
 p=('../persistent/index_data/' if x=='DXY' else '../persistent/stock_data/')+x+'.csv'; return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().close.astype(float)
px=pd.DataFrame({x:s(x) for x in U}).loc[:'2026-07-15']; r=px.pct_change();
# volatility compression: negative 20d realized vol, residualized cross-sectionally against 60d vol
v=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=45).std(); f=-(v/v60)
for h in [1,5,10]:
 a=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(a); print('H',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'names',f.notna().sum(axis=1).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for yr in range(2020,2027):
 a=[]
 for i in range(len(px)-1):
  if px.index[i].year!=yr:continue
  z=pd.concat([f.iloc[i],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 if a:print('REG',yr,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1))
print('period',px.index.min(),px.index.max())
