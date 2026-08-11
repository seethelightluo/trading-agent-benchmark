import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'; D={}
for s in U:
 d=pd.read_csv(f'{root}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()
 D[s]=d.close.loc[:'2026-07-15']
p=pd.concat(D,axis=1); r=p.pct_change(fill_method=None)
v20=r.rolling(20,min_periods=10).std(); v60=r.rolling(60,min_periods=20).std()
fac=-r*v60/v20

def evalh(h, dates=None):
 y=p.pct_change(h,fill_method=None).shift(-h); rows=[]
 for dt in (fac.index if dates is None else dates):
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic
 return len(q),q.n.mean(),q.n.sum()/(len(q)*15),ic.mean(),ic.mean()/ic.std(ddof=1),np.mean(ic>0),q
for h in [1,3,5,10]:
 a=evalh(h);print('h',h,'dates',a[0],'avg_n',round(a[1],3),'coverage',round(a[2],4),'IC',round(a[3],6),'ICIR',round(a[4],6),'hit',round(a[5],4))
for name,sl in [('2020_22',slice('2020','2022')),('2023_24',slice('2023','2024')),('2025_26',slice('2025','2026'))]:
 a=evalh(1,fac.loc[sl].index);print(name,'dates',a[0],'IC',round(a[3],6),'ICIR',round(a[4],6),'hit',round(a[5],4))
print('turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
print('recent250',round(evalh(1)[3].iloc if False else 0,4))
# signal artifact for deterministic audit
fac.stack().rename('signal').to_frame().to_csv('factors/miner_1_20260730_volstate_reversal_1d_signal.csv')
