import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-10-10')
def rd(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); d=d[d.index<=E]; return d[['open','close','volume']].apply(pd.to_numeric,errors='coerce')
D={a:rd(a) for a in A}; O=pd.DataFrame({a:D[a]['open'] for a in A}); C=pd.DataFrame({a:D[a]['close'] for a in A}); V=pd.DataFrame({a:D[a]['volume'] for a in A})
# Intraday-vs-overnight efficiency: assets with persistently weak intraday closes,
# but unusually strong overnight gaps, are candidates for next-period intraday catch-up.
gap=O/C.shift(1)-1; intra=C/O-1; vol=C.pct_change().rolling(20,min_periods=12).std()
# lagged, volatility-scaled differential of overnight and intraday returns
F=((gap-intra).rolling(10,min_periods=6).mean()/(vol+1e-12)).shift(1)
F=F.sub(F.mean(axis=1),axis=0); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('idea=overnight-intraday dislocation catchup rows',len(C),'assets',len(A),'valid_cells',int(F.notna().sum().sum()),'coverage',round(F.notna().mean().mean(),4))
def test(h):
 fw=C.shift(-h)/C-1; out=[]; dates=[]; nn=[]
 for t in F.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.r.nunique()>1: out.append(q.f.corr(q.r,method='spearman')); dates.append(t); nn.append(len(q))
 x=pd.Series(out,index=dates); print('H',h,'dates',len(x),'meanN',round(np.mean(nn),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4)); return x
X={h:test(h) for h in [1,5,10,20]}
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-10-10')]:
 x=X[10][(X[10].index>=lo)&(X[10].index<=hi)]; print('regime',lo,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
# elementary overlap diagnostics only; exact library histories are not encoded in JSON files
for n,x in {'gap10':gap.rolling(10,min_periods=6).mean(),'intra10':intra.rolling(10,min_periods=6).mean(),'vol20':vol}.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); print('proxy',n,'rho',round(q.f.corr(q.x,method='spearman'),6),'cells',len(q))
print('LIBRARY_AUDIT=FAILED exact aligned signal histories for all admitted factors unavailable; no persistence permitted')
