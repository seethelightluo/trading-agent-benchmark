import numpy as np,pandas as pd,glob,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 fs=glob.glob('../persistent/stock_data/'+s+'.csv')
 if not fs: continue
 d=pd.read_csv(fs[0]); d['date']=pd.to_datetime(d.date); d=d.sort_values('date')
 if len(d)<420: continue
 c=d.close.astype(float); r=c.pct_change(); loc=((c-d.low)/(d.high-d.low).replace(0,np.nan)-.5)
 pos=loc.rolling(25,min_periods=12).mean(); tr=c.pct_change(60); v=r.rolling(90,min_periods=45).std()*np.sqrt(60)
 f=(tr/(v+1e-8))*pos
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor':f.shift(1),'close':c}))
x=pd.concat(rows,ignore_index=True).sort_values(['symbol','date'])
os.makedirs('scripts/artifacts',exist_ok=True); x[['date','symbol','factor']].to_csv('scripts/artifacts/miner_3_20330303_range_slow_trend_signal.csv',index=False)
for H in [10,20,30,40]:
 z=x.copy(); z['fwd']=z.groupby('symbol').close.shift(-H)/z.close-1; a=[];ns=[]
 for _,g in z.groupby('date'):
  g=g.dropna(subset=['factor','fwd'])
  if len(g)>=8:
   q=g.factor.corr(g.fwd,method='spearman')
   if pd.notna(q):a.append(q);ns.append(len(g))
 q=pd.Series(a); print(f'H{H} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.8f} ICIR={q.mean()/(q.std(ddof=1)+1e-12):.6f} hit={(q>0).mean():.4f}')
for n in [260,520,780]:
 q=x.copy(); q['fwd']=q.groupby('symbol').close.shift(-30)/q.close-1; cutoff=q.date.max()-pd.Timedelta(days=n*1.45); q=q[q.date>=cutoff]; a=[]
 for _,g in q.groupby('date'):
  g=g.dropna(subset=['factor','fwd'])
  if len(g)>=8:
   v=g.factor.corr(g.fwd,method='spearman')
   if pd.notna(v):a.append(v)
 s=pd.Series(a); print(f'recent{n} dates={len(s)} IC={s.mean():.8f} ICIR={s.mean()/(s.std(ddof=1)+1e-12):.6f}')
print('symbols',x.symbol.nunique(),'dates',x.date.nunique(),'coverage',x.factor.notna().mean());x['rank']=x.groupby('date')['factor'].rank(pct=True); x['pr']=x.groupby('symbol')['rank'].shift(1); print('turnover',(x['rank']-x.pr).abs().mean())
