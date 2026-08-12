import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acct=get_account_dict(); syms=acct.get('watch_list',[])
if not syms: syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in syms:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d)>80:
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); prices[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(prices).sort_index(); r=p.pct_change()
# equal-weight market return, rolling beta and conditional defensive beta signal
m=r.mean(axis=1,skipna=True)
rows=[]
for dt in p.index:
    ix=p.index.get_loc(dt)
    if ix<25: continue
    hist=r.iloc[max(0,ix-20):ix] # completed bars through previous date when dt is signal date
    valid=hist.notna().sum(); cov=hist.apply(lambda z: z.cov(m.loc[z.index]),axis=0)
    var=m.loc[hist.index].var()
    beta=cov/var if var>1e-10 else cov*0
    # defensive low-beta, additionally stress conditioned by negative market 5d return
    market5=(p.iloc[ix-5:ix].pct_change().mean(axis=1).add(1).prod()-1) if ix>=6 else 0
    sig=-beta
    if market5 < 0: sig=sig*1.5
    for s in syms:
        if s in p.columns and pd.notna(sig.get(s)) and ix+1<len(p) and pd.notna(r.iloc[ix+1].get(s)):
            rows.append((dt,s,float(sig[s]),float(r.iloc[ix+1][s])))
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).set_index('date')
ics=df.groupby(level=0).apply(lambda z:z.factor.corr(z.fwd) if len(z)>=8 else np.nan).dropna()
print('dates',len(ics),'symbols',len(syms),'avg_n',df.groupby(level=0).size().mean(),'coverage',len(df)/(len(ics)*len(syms)))
print('h1 IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1), (ics>0).mean()))
for lag in [5,10]:
    # recompute forward multi-day based on prices at dt and dt+lag; simple on aligned panel
    arr=[]
    for dt,g in df.groupby(level=0):
      if dt not in p.index: continue
      i=p.index.get_loc(dt)
      if i+lag>=len(p): continue
      for _,z in g.iterrows():
       s=z.symbol
       if s in p and pd.notna(p.iloc[i+lag][s]) and pd.notna(p.iloc[i][s]): arr.append((dt,z.factor,p.iloc[i+lag][s]/p.iloc[i][s]-1))
    q=pd.DataFrame(arr,columns=['date','factor','fwd']).groupby('date').apply(lambda z:z.factor.corr(z.fwd) if len(z)>=8 else np.nan).dropna()
    print('h%d IC %.6f ICIR %.6f hit %.4f dates %d'%(lag,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
# signal artifact
out=df.reset_index()[['date','symbol','factor']]; out.to_csv('scripts/miner_1_20300725_defensive_beta_signal.csv',index=False)
