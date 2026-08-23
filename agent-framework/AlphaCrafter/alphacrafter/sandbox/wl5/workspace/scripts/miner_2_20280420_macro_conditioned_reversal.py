import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); scale=(p.diff()/p.shift()).abs().rolling(20).median()
base=-(r.rolling(5).sum()/(scale*np.sqrt(5)+1e-12)); br=base.rank(axis=1,pct=True).sub(.5)
v=get_index_daily_data('VIX',days=2400)
if v is None or len(v)==0: raise RuntimeError('VIX unavailable')
v=v.set_index('date')['close'].reindex(p.index).ffill()
high=(v>v.rolling(60,min_periods=30).median()).astype(float).shift(1)
trend=(v.pct_change(10)<0).astype(float).shift(1)
variants={'base':br,'lowvol_reversal':br.mul(1-high*.65,axis=0),'vix_falling_reversal':br.mul(.55+.45*trend,axis=0),'combined':br.mul((1-high*.55)*(.65+.35*trend),axis=0)}
rows=[]
for name,f in variants.items():
 for i in range(len(p)-10):
  x=f.iloc[i]; fr=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&fr.notna()
  if ok.sum()>=8: rows.append((name,p.index[i],x[ok].corr(fr[ok]),ok.mean()))
df=pd.DataFrame(rows,columns=['factor','date','ic','coverage']).dropna()
for name in variants:
 z=df[df.factor==name]
 print(name,'dates',len(z),'avg_names',round(z.coverage.mean()*len(U),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4),'turnover',round(variants[name].diff().abs().mean().mean(),6))
 for lab,q in [('2020_24',z[z.date<'2025-01-01']),('2025_26',z[(z.date>='2025-01-01')&(z.date<'2027-01-01')]),('2027_28',z[z.date>='2027-01-01'])]:
  print(' ',lab,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else np.nan)
variants['combined'].reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_2_20280420_macro_conditioned_reversal_signal.csv',index=False)
