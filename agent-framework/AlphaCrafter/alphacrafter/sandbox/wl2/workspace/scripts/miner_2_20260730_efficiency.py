import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# per-asset rolling calculations preserve each asset's own trading calendar
F={}; Y={h:{} for h in [1,5,10]}
for s,x in D.items():
 r=x.close.pct_change(); F[s]=(x.close/x.close.shift(20)-1)/r.abs().rolling(20).sum().replace(0,np.nan)
 for h in Y: Y[h][s]=x.close.shift(-h)/x.close-1
F=pd.DataFrame(F); Y={h:pd.DataFrame(v) for h,v in Y.items()}
def test(y):
 ic=[]; ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':y.loc[dt]}).dropna()
  if len(z)>=8: ic.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 a=np.asarray(ic); return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),(a>0).mean()
for h in [1,5,10]: print(h,test(Y[h]))
for yr in range(2020,2027):
 a=[]
 for dt in F.loc[str(yr)].index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y[1].loc[dt]}).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
 print(yr,len(a),round(np.mean(a),5) if a else None)
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
