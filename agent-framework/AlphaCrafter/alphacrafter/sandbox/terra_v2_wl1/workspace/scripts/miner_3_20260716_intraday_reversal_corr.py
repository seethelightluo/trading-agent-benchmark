import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# candidate: prior completed session intraday reversal, known at close and predicts next close
F={}; existing={}
for s,x in D.items():
 F[s]=-(x.close/x.open-1)
 existing[s]=pd.DataFrame({'clv':-(2*(x.close-x.low)/(x.high-x.low).replace(0,np.nan)-1),'rev5':-(x.close/x.close.shift(5)-1),'mom20':x.close/x.close.shift(20)-1})
f=pd.concat(F,axis=1).stack().rename('f'); e=pd.concat(existing,axis=0); e.index.names=['symbol','date']; f.index.names=['date','symbol']; f=f.reorder_levels(['symbol','date']).sort_index()
for c in e.columns: print(c, f.corr(e[c]))
# full validation
for h in [1,5,10]:
 out=[]
 for s,x in D.items():
  z=pd.DataFrame({'f':F[s],'r':x.close.shift(-h)/x.close}).dropna()
  for dt,g in z.groupby(z.index):
   if len(g)>=8: out.append(g.f.corr(g.r,method='spearman'))
 a=np.array(out); print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
print('coverage',np.mean([F[s].notna().mean() for s in U]))
# rank turnover proxy
print('turnover',np.mean([F[s].rank(pct=True).diff().abs().mean() for s in U]))
