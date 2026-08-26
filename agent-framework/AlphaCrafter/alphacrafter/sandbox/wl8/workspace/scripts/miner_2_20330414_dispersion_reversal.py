import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}
px=pd.DataFrame(D).loc[:'2033-04-13']
r5=px.pct_change(5); r20=px.pct_change(20); r1=px.pct_change()
disp=r1.sub(r1.mean(axis=1),axis=0).abs().mean(axis=1).rolling(5).mean()
z=(disp-disp.rolling(120).mean())/disp.rolling(120).std()
g=np.tanh(z).fillna(0)
variants={'disp_reversal_5d':-r5.mul(1+.8*g,axis=0),'disp_reversal_10d':-r20.mul(1+.8*g,axis=0),'disp_relative_reversal':-(r5.sub(r5.mean(axis=1),axis=0)).mul(1+.8*g,axis=0),'disp_reversal_blend':-(.7*r5+.3*r20).mul(1+.8*g,axis=0)}
def icrows(a,b):
 ok=a.notna()&b.notna(); n=ok.sum(axis=1)
 ar=a.rank(axis=1); br=b.rank(axis=1)
 am=ar.where(ok).mean(axis=1); bm=br.where(ok).mean(axis=1)
 cov=((ar-am.values[:,None])*(br-bm.values[:,None])).where(ok).sum(axis=1)
 sd1=np.sqrt(((ar-am.values[:,None])**2).where(ok).sum(axis=1)); sd2=np.sqrt(((br-bm.values[:,None])**2).where(ok).sum(axis=1))
 return (cov/(sd1*sd2)).where(n>=8)
for name,f in variants.items():
 b=px.shift(-10)/px-1; q=icrows(f,b).dropna(); turn=(f.rank(axis=1).diff().abs().mean(axis=1)/15).mean()
 print(name,'dates',len(q),'avgN',((f.notna()&b.notna()).sum(axis=1)).mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean(),'turn',turn)
 print('decay',[(h,round(icrows(f,px.shift(-h)/px-1).dropna().mean(),6)) for h in [1,5,10,20]])
