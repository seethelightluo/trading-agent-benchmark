import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; xs={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<250:d=get_index_daily_data(s,days=4000)
 if d is not None and len(d)>250:
  d=d.copy();d['date']=pd.to_datetime(d.date);xs[s]=d.set_index('date').close.astype(float).pct_change()
r=pd.DataFrame(xs).sort_index(); v=r.rolling(20,min_periods=15).std(); x=r.rolling(3,min_periods=3).sum()/v
# residualize against cross-sectional mean and smooth
f=-x.sub(x.mean(axis=1),axis=0).ewm(span=3,min_periods=2).mean()
rows=[]; turns=[]
for i in range(len(r)-10):
 a=f.iloc[i]; ok=a.notna()&r.iloc[i+1:i+6].sum().notna(); fr=r.iloc[i+1:i+6].sum()
 if ok.sum()>=8:rows.append((r.index[i],ok.sum(),a[ok].corr(r.iloc[i+1][ok]),a[ok].corr(fr[ok])))
 if i>2:
  q=f.iloc[i].rank(pct=True);p=f.iloc[i-1].rank(pct=True);turns.append((q-p).abs().mean())
z=pd.DataFrame(rows,columns=['date','n','ic1','ic5']).set_index('date')
def st(a):
 a=a.dropna();return len(a),round(a.mean(),5),round(a.std(),5),round(a.mean()/a.std(),5),round((a>0).mean(),4)
print('dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.sum()/(len(z)*15),4),'turnover',round(np.nanmean(turns),4))
for c in ['ic1','ic5']:
 print(c,st(z[c]))
 for l,s in [('20-22',z.loc[:'2022']),('23-25',z.loc['2023':'2025']),('26-28',z.loc['2026':'2028']),('29+',z.loc['2029':])]:print(l,st(s[c]))
f.to_csv('scripts/miner_1_20300207_short_reversal_signal.csv',index_label='date')
