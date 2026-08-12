import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s, days=3600)
    except Exception: x=get_stock_daily_data(s, days=3600)
    if x is not None and len(x)>150:
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').drop_duplicates('date'); x['r']=np.log(x.close).diff(); D[s]=x.set_index('date')
# candidate: recovery-adjusted trend = 20d log return / downside deviation(60d), multiplied by 20d up-day share
rows=[]
for s,x in D.items():
    r=x.r
    mom=r.rolling(20).sum()
    down=np.sqrt((r.clip(upper=0)**2).rolling(60).mean())
    up=(r>0).rolling(20).mean()
    sig=(mom/(down+1e-8))*up
    fwd=np.log(x.close).shift(-1)-np.log(x.close)
    for d in x.index:
        if pd.notna(sig.loc[d]) and pd.notna(fwd.loc[d]): rows.append((d,s,float(sig.loc[d]),float(fwd.loc[d])))
a=pd.DataFrame(rows,columns=['date','symbol','sig','fwd'])
# cross-sectional spearman IC; minimum 8
out=[]
for d,g in a.groupby('date'):
    if len(g)>=8: out.append((d,g.sig.rank().corr(g.fwd.rank()),len(g),g.sig.notna().mean()))
o=pd.DataFrame(out,columns=['date','ic','n','cov']).dropna()
print('candidate=recovery_adjusted_trend20; dates',len(o),'avg_names',o.n.mean(),'coverage',a.sig.notna().mean())
print('1d meanIC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1)*np.sqrt(len(o)),(o.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2029')]:
 z=o[(o.date>=lo)&(o.date<=hi+'-12-31')]
 print(lo,hi,len(z),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(len(z)),4))
for h in [3,5,10]:
 rr=[]
 for s,x in D.items():
  r=np.log(x.close).shift(-h)-np.log(x.close)
  mom=x.r.rolling(20).sum(); down=np.sqrt((x.r.clip(upper=0)**2).rolling(60).mean()); up=(x.r>0).rolling(20).mean(); sig=mom/(down+1e-8)*up
  for d in x.index:
   if pd.notna(sig.loc[d]) and pd.notna(r.loc[d]): rr.append((d,s,sig.loc[d],r.loc[d]))
 z=pd.DataFrame(rr,columns=['date','s','sig','fwd']); q=[]
 for d,g in z.groupby('date'):
  if len(g)>=8:q.append(g.sig.rank().corr(g.fwd.rank()))
 q=pd.Series(q).dropna(); print(h,'d meanIC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q))))
# save raw signal for audit
# use latest available signal per symbol
latest=[]
for s,x in D.items():
 sig=x.r.rolling(20).sum()/(np.sqrt((x.r.clip(upper=0)**2).rolling(60).mean())+1e-8)*(x.r>0).rolling(20).mean()
 for d,v in sig.dropna().tail(2500).items(): latest.append({'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)})
pd.DataFrame(latest).to_csv('scripts/miner_2_20290712_recovery_trend20_signal.csv',index=False)
# turnover rank proxy
p=a.pivot(index='date',columns='symbol',values='sig').rank(axis=1,pct=True); print('turnover_proxy',p.diff().abs().mean(axis=1).mean())
