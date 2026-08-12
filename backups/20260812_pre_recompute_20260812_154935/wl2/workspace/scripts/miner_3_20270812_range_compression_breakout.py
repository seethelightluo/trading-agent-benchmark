import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2027-08-11'); dates=D['SPX'].index[(D['SPX'].index>='2020-02-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U})
r=C.pct_change(); atr=(H-L).rolling(20).mean()/C
# range-compression breakout: medium return, penalized by recent range, with a short-term shock cap
F=(C.pct_change(10)/(atr* np.sqrt(10))).clip(-3,3).shift(1)
y=C.shift(-1).div(C)-1
A=[];ds=[];ns=[]
for dt in dates:
 z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.f,z.y).statistic
  if np.isfinite(q):A.append(q);ds.append(dt);ns.append(len(z))
a=np.array(A)
def out(x): return (round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)),6)) if len(x)>1 else (None,None)
print('factor range_compression_breakout_10d dates',len(a),'avgN',round(np.mean(ns),2),'IC/ICIR',out(a),'hit',round((a>0).mean(),4))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 z=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(z),'IC/ICIR',out(z))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for h in [3,5,10]:
 yy=C.shift(-h).div(C)-1;aa=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),yy.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:aa.append(spearmanr(z.f,z.y).statistic)
 aa=np.array(aa);print('horizon',h,'dates',len(aa),'IC/ICIR',out(aa))
print('signal_artifact',F.tail(1).to_json())
