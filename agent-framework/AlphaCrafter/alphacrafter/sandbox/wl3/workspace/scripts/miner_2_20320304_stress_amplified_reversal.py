import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
dates=sorted(set.intersection(*[set(s.index) for s in px.values()]))
# Candidate: stress-amplified short-term reversal, with volatility normalization.
rows=[]; daily=[]
for i,d in enumerate(dates):
    if i<30 or i+10>=len(dates): continue
    vals={}; fwd={}
    for a in assets:
        s=px[a]
        if d not in s.index: continue
        hist=s.loc[:d]
        if len(hist)<21: continue
        r3=np.log(hist.iloc[-1]/hist.iloc[-4])
        vol=hist.pct_change().rolling(20).std().iloc[-1]
        if not np.isfinite(vol) or vol<=0: continue
        vals[a]=-r3/vol
        # align forward by common dates
        try: fwd[a]=np.log(s.loc[dates[i+10]]/s.loc[d])
        except: pass
    # VIX state available at d; use normalized recent shock, clipped
    vh=vix.loc[:d].dropna()
    if len(vh)<21: continue
    shock=(vh.iloc[-1]/vh.iloc[-11]-1)
    mult=1+0.75*np.clip(shock,-1,2)
    vals={a:x*mult for a,x in vals.items() if a in fwd}
    if len(vals)>=8:
        ic=spearmanr(list(vals.values()),[fwd[a] for a in vals]).statistic
        daily.append((d,ic,len(vals)))
        rows += [(d,a,vals[a],fwd[a]) for a in vals]
df=pd.DataFrame(daily,columns=['date','ic','n'])
print('dates',len(df),'avgN',df.n.mean(),'coverage',df.n.mean()/15,'IC',df.ic.mean(),'ICIR',df.ic.mean()/df.ic.std(ddof=1),'hit', (df.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2032')]:
 x=df[(df.date.dt.year>=int(lo))&(df.date.dt.year<=int(hi))].ic
 print(lo,hi,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
print('recent60',df.ic.tail(60).mean(),'recent120',df.ic.tail(120).mean())
out='scripts/miner_2_20320304_stress_amplified_reversal'; pd.DataFrame(rows,columns=['date','asset','signal','fwd10']).to_csv(out+'_signal.csv',index=False); df.to_csv(out+'_ic.csv',index=False)
