import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).reset_index(drop=True).sort_values('date').loc[lambda z:z.date<='2026-07-15'] for s in U}
def metrics(expr):
 allx=[]
 for s,x in D.items(): allx.append(pd.DataFrame({'date':x.date.values,'symbol':s,'f':expr(x).values,'r':(x.close.shift(-1)/x.close-1).values}))
 z=pd.concat(allx,ignore_index=True).dropna(); out=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: out.append((dt,g.f.rank().corr(g.r.rank()),len(g)))
 q=pd.DataFrame(out,columns=['date','ic','n']); a=q.ic.to_numpy()
 return len(q),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),q
names={'intraday_reversal':lambda x: -(x.close/x.open-1),'clv':lambda x: 2*(x.close-x.low)/(x.high-x.low).replace(0,np.nan)-1,'rev5':lambda x: -(x.close/x.close.shift(5)-1),'mom20':lambda x: x.close/x.close.shift(20)-1}
res={}
for n,e in names.items():
 k,ic,ir,hit,q=metrics(e); res[n]=q.set_index('date').ic
 print(n,'dates',k,'IC',round(ic,5),'ICIR',round(ir,5),'hit',round(hit,4))
base=res['intraday_reversal']
for n,v in res.items():
 if n!='intraday_reversal': print('corr',n,round(base.corr(v),5))
print('coverage',np.mean([names['intraday_reversal'](x).notna().mean() for x in D.values()]))
