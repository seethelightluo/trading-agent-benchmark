import numpy as np, pandas as pd, glob
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 fs=glob.glob('../persistent/stock_data/'+s+'.csv')
 if not fs: continue
 d=pd.read_csv(fs[0]); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date')
 if len(d)<350: continue
 c=d.close.astype(float); h=d.high.astype(float); l=d.low.astype(float)
 r=c.pct_change(); loc=((c-l)/(h-l).replace(0,np.nan)-.5)
 persistence=loc.rolling(10,min_periods=6).mean(); trend=c.pct_change(30)
 vol=r.rolling(60,min_periods=30).std()*np.sqrt(30)
 consistency=(r.gt(0).rolling(30,min_periods=20).mean()-.5)*2
 f=(trend/(vol+1e-8))*persistence.abs().clip(0,1)*(0.5+0.5*consistency.abs())*np.sign(trend)
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor':f.shift(1),'close':c}))
x=pd.concat(rows,ignore_index=True).sort_values(['symbol','date'])
for H in [5,10,20,30]:
 z=x.copy(); z['fwd']=z.groupby('symbol').close.shift(-H)/z.close-1; vals=[]; ns=[]
 for _,g in z.groupby('date'):
  g=g.dropna(subset=['factor','fwd'])
  if len(g)>=8:
   q=g.factor.corr(g.fwd,method='spearman')
   if pd.notna(q): vals.append(q); ns.append(len(g))
 q=pd.Series(vals); print(f'H{H} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.8f} ICIR={q.mean()/(q.std(ddof=1)+1e-12):.6f} hit={(q>0).mean():.4f}')
print('symbols',x.symbol.nunique(),'dates',x.date.nunique(),'coverage',x.factor.notna().mean())
x['rank']=x.groupby('date')['factor'].rank(pct=True); x['prev_rank']=x.groupby('symbol')['rank'].shift(1)
print('turnover', (x['rank']-x['prev_rank']).abs().mean())
