import pandas as pd, numpy as np

ASOF = '2026-07-29'
assets = ['000300.SH','000688.SH','SPX','NDX','SOX','HSI','N225','SX5E',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
obs = ['VIX','DXY','USDCNY','USDJPY','EURUSD']

def load(p):
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= ASOF].reset_index(drop=True)
    return df

rows = []
for a in assets:
    df = load(f'../persistent/stock_data/{a}.csv')
    c = df['close']
    last = c.iloc[-1]
    def ret(n):
        return c.iloc[-1]/c.iloc[-1-n]-1 if len(c)>n else np.nan
    vol20 = c.pct_change().tail(20).std()*np.sqrt(252)
    ma20 = c.rolling(20).mean().iloc[-1]; ma60 = c.rolling(60).mean().iloc[-1]
    dd = (c.tail(60)/c.tail(60).cummax()-1).min()
    up = (c.pct_change().tail(20)>0).mean()
    rows.append(dict(asset=a, last=round(last,2), r20=round(ret(20)*100,1),
                     r40=round(ret(40)*100,1), r60=round(ret(60)*100,1),
                     vol20=round(vol20*100,1), ma_slope=round((ma20/ma60-1)*100,1),
                     dd60=round(dd*100,1), up20=round(up*100,0)))
t = pd.DataFrame(rows)
print(t.to_string(index=False))
print()
for o in obs:
    df = load(f'../persistent/index_data/{o}.csv')
    c = df['close']
    last = c.iloc[-1]
    def ret(n):
        return c.iloc[-1]/c.iloc[-1-n]-1 if len(c)>n else np.nan
    print(f"{o}: last={last:.2f} r20={ret(20)*100:.1f}% r40={ret(40)*100:.1f}% r60={ret(60)*100:.1f}%")
