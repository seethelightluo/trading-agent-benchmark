import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    try: x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
    except: continue
    rng=(x.high-x.low).replace(0,np.nan)
    clv=(2*x.close-x.high-x.low)/rng
    vr=(x.volume/x.volume.rolling(20,min_periods=10).median()).clip(0,3)
    f=-(clv*vr).rolling(3,min_periods=3).mean()
    for d,v in zip(x.date,f):
        if pd.notna(v): rows.append({'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)})
pd.DataFrame(rows).to_csv('scripts/miner_2_20271103_volume_pressure_reversal_3d_signal.csv',index=False)
print('rows',len(rows),'symbols',pd.DataFrame(rows).symbol.nunique())
