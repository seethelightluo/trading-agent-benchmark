import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].sort_index().loc[:cut].reindex(P.index).ffill()
r20=P.pct_change(20); vlevel=v.rolling(60,min_periods=40).mean(); vratio=v/vlevel-1
mult=(1-vratio.clip(-.5,.5)).clip(.5,1.5); f=r20.mul(mult,axis=0)
def run(h):
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('horizon',h,'dates',len(ic),'avg_n',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 return ic,a
ic,a=run(1)
for yr,g in ic.groupby(ic.index.year): print('regime',yr,len(g),round(g.mean(),6),round(g.mean()/g.std(ddof=1),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for h in [5,10]: run(h)
z=pd.concat([f.stack().rename('f'),r20.stack().rename('r20')],axis=1).dropna();print('corr_r20',round(z['f'].corr(z['r20']),6))
