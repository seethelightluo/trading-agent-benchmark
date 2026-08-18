import numpy as np,pandas as pd,json
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2032-05-13')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close[d.index<=cutoff]
px=pd.DataFrame(P); r=px.pct_change(); cs=r.rolling(20,min_periods=15).std(); ret=r.rolling(10,min_periods=10).sum(); breadth=(r>0).rolling(20,min_periods=15).mean().mean(axis=1)
# contrarian relative return, active in weak/strong breadth extremes, volatility scaled
rel=ret.sub(ret.median(axis=1),axis=0); fac=-rel/(cs*np.sqrt(10)+1e-9); gate=((breadth<.40)|(breadth>.67)); fac=fac.where(gate)
def run(h):
 vals=[]; turns=[]; prev=None
 for i in range(len(px)-h):
  z=pd.DataFrame({'f':fac.iloc[i],'r':px.pct_change(h).shift(-h).iloc[i]}).dropna(); z=z[z.f.replace([np.inf,-np.inf],np.nan).notna()]
  if len(z)>=8 and z.f.nunique()>1:
   vals.append((px.index[i],z.f.corr(z.r,method='spearman'),len(z)))
   q=z.f.rank(pct=True); turns.append(np.mean(abs(q-(prev.reindex(q.index) if prev is not None else q)))) if prev is not None else None; prev=q
 a=np.array([v[1] for v in vals]); return {'dates':len(a),'avg_n':float(np.mean([v[2] for v in vals])),'coverage':float(np.sum([v[2] for v in vals])/(len(a)*15)),'ic':float(np.mean(a)),'icir':float(np.mean(a)/(np.std(a,ddof=1)+1e-12)*np.sqrt(252)),'hit':float(np.mean(a>0)),'turnover':float(np.mean(turns)) if turns else None,'recent365':float(np.mean([v[1] for v in vals if v[0]>=cutoff-pd.Timedelta(days=365)]))}
for h in [5,10,20]: print(h,json.dumps(run(h)))
