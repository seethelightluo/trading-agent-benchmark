import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 if len(d)>100: px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().ffill().loc[:'2026-07-15']
r5=p.pct_change(5); r3=p.pct_change(3)
# Leave-one-out peer impulse, modulated by cross-sectional agreement/breadth.
peer=pd.DataFrame({s:r5.drop(columns=s).median(axis=1) for s in px})
breadth=(r3>0).sum(axis=1)/len(px)
agreement=(breadth-0.5).abs()*2
# signed agreement: broad up rewards positive peer impulse, broad down rewards negative impulse
signed=2*breadth-1
f=peer.mul(0.5+0.5*agreement,axis=0) # magnitude confidence, retains direction
f2=peer.mul(signed,axis=0) # conditional alignment with market breadth
future=p.pct_change().shift(-1); future5=p.pct_change(5).shift(-5)
def run(z,label):
 obs=[]; ic5=[]; ns=[]; ranks=[]
 for dt in z.index:
  q=pd.concat([z.loc[dt].rename('f'),future.loc[dt].rename('y'),future5.loc[dt].rename('y5')],axis=1).dropna()
  if len(q)>=8:
   obs.append(spearmanr(q.f,q.y).statistic); ic5.append(spearmanr(q.f,q.y5).statistic); ns.append(len(q)); ranks.append(q.f.rank(pct=True))
 a=np.asarray(obs); b=np.asarray(ic5)
 print(label,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4),'5dIC',round(np.nanmean(b),6),'5dICIR',round(np.nanmean(b)/np.nanstd(b,ddof=1),6),'coverage',round(np.mean(ns)/len(U),4))
run(f,'confidence_peer')
run(f2,'breadth_signed_peer')
# pooled rank correlations versus existing proxy factors
for name,z in [('candidate',f),('candidate2',f2)]:
 for nm,q in [('rev5',-r5),('mom20',p.pct_change(20)/p.pct_change(20).rolling(20).std()),('peer',peer)]:
  x=pd.concat([z.stack().rename('z'),q.stack().rename('q')],axis=1).dropna(); print('corr',name,nm,round(x.z.corr(x.q),4))
# regime IC for best
for period,sl in [('2020-22',('2020','2022-12-31')),('2023-24',('2023','2024-12-31')),('2025-26',('2025','2026-07-15'))]:
 o=[]
 for dt in f.loc[sl[0]:sl[1]].index:
  q=pd.concat([f.loc[dt].rename('f'),future.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8:o.append(spearmanr(q.f,q.y).statistic)
 print('regime',period,'n',len(o),'IC',round(np.nanmean(o),6),'ICIR',round(np.nanmean(o)/np.nanstd(o,ddof=1),6) if len(o)>1 else None)
