import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
acct=get_account_dict(); universe=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in universe:
    d=None
    try: d=get_index_daily_data(s, days=3000)
    except Exception: pass
    if d is None:
        try: d=get_stock_daily_data(s, days=3000)
        except Exception: pass
    if d is not None and len(d)>0:
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.dropna().drop_duplicates('date').sort_values('date'); frames[s]=x.set_index('date').close
px=pd.DataFrame(frames).sort_index().ffill(); rets=px.pct_change()
vol=rets.rolling(30,min_periods=25).std()*np.sqrt(252)
sig=(1/(vol+0.01)).shift(1).rank(axis=1,pct=True)
for h in [1,5,10,20]:
    fwd=px.shift(-h)/px-1; ics=[]; dates=[]; counts=[]
    for dt in sig.index:
        z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); counts.append(len(z))
    v=np.array([x for x in ics if np.isfinite(x)])
    ir=v.mean()/v.std(ddof=1)*np.sqrt(len(v)) if len(v)>1 and v.std(ddof=1)>0 else 0
    print('H',h,'dates',len(v),'avg_n',round(float(np.mean(counts)),2),'IC',round(float(v.mean()),6),'ICIR',round(float(ir),6),'hit',round(float((v>0).mean()),4))
    if h==10:
        rr=sig; turns=[]
        for i in range(1,len(rr)):
            common=rr.iloc[i].dropna().index.intersection(rr.iloc[i-1].dropna().index)
            if len(common)>=8: turns.append(np.mean(np.abs(rr.iloc[i][common]-rr.iloc[i-1][common])))
        print('TURN',round(float(np.mean(turns)),6),'coverage',round(float(sig.notna().sum().sum()/(sig.shape[0]*len(universe))),4),'universe',len(universe),'assets_loaded',len(frames))
        for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31')]:
            q=np.array([ics[i] for i,d in enumerate(dates) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi) and np.isfinite(ics[i])]); qi=q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 and q.std(ddof=1)>0 else 0
            print('REG',label,'n',len(q),'IC',round(float(q.mean()),6) if len(q) else None,'ICIR',round(float(qi),6) if len(q) else None)
print('range',px.index.min(),px.index.max(),'assets',list(frames))

# deterministic provenance artifact
sig.to_csv("scripts/miner_3_20280703_lowvol_defensive_signal.csv", index_label="date")
