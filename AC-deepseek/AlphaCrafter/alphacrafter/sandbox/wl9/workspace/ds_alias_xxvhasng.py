import pandas as pd, numpy as np
from pathlib import Path
CUR='2029-08-08'
assets=['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','WTI','XAU','US10Y']
px={}
for a in assets:
    df=pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date']=pd.to_datetime(df['date'])
    df=df[df['date']<=CUR].set_index('date').sort_index()
    px[a]=df['close']
PX=pd.DataFrame(px)
# trailing 90d returns
ret=(PX.pct_change().tail(90))
cum=(1+ret).cumprod().iloc[-1]-1
print("=== 90d cumulative return to",CUR,"===")
print(cum.sort_values().to_string())
print("\n=== 20d return ===")
r20=(1+PX.pct_change().tail(20)).cumprod().iloc[-1]-1
print(r20.sort_values().to_string())
# vol proxy: 20d realized vol annualized
vol=PX.pct_change().tail(20).std()*np.sqrt(252)
print("\n=== 20d annualized vol ===")
print(vol.sort_values().round(2).to_string())