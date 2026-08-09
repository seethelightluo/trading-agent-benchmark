import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]; R=P.pct_change()
# cross-asset defensive quality: inverse realized volatility, with winsorization by date
vol=R.rolling(20,min_periods=15).std(); f=1/vol
# rank makes scale comparable and avoids outliers
f=f.rank(axis=1,pct=True)
def go(h):
 y=P.shift(-h).div(P)-1; z=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(z,columns=['d','ic','n']).set_index('d'); x=a.ic
 print(h,len(x),round(a.n.mean(),2),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
 return x
x=go(1)
for yr,g in x.groupby(x.index.year):print(yr,round(g.mean(),6),round(g.mean()/g.std(ddof=1),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.diff().abs().mean().mean(),4))
for h in [5,10]:go(h)
