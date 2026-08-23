import pandas as pd, json, numpy as np
cut = pd.to_datetime(json.load(open('../persistent/date.json'))['visible_through'])
WATCHLIST = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
closes = {}
for s in WATCHLIST:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date']<=cut].sort_values('date').reset_index(drop=True)
    closes[s] = df.set_index('date')['close'].astype(float)
panel = pd.DataFrame(closes).sort_index()
for h in [5,10,21,63]:
    r = panel.iloc[-1]/panel.iloc[-1-h]-1
    print(f"ret_{h}d: ", {k: round(v,4) for k,v in r.items()})
print()
vix = pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); vix=vix[vix['date']<=cut].sort_values('date').reset_index(drop=True)
v = vix.set_index('date')['close'].astype(float)
print('VIX last 15 closes:', [round(x,1) for x in v.tail(15).values])
print('VIX 60d ago:', round(float(v.iloc[-61]),1), ' 21d ago:', round(float(v.iloc[-22]),1), ' 10d ago:', round(float(v.iloc[-11]),1))