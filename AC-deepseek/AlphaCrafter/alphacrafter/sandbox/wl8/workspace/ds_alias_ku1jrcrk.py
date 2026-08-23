import pandas as pd, json, os
cut = pd.to_datetime(json.load(open('../persistent/date.json'))['visible_through'])
print('visible_through:', cut.date())
print()
print(f"{'symbol':<10}{'rows':>6}{'last_date':>14}{'last_close':>12}")
for s in ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
    df['date'] = pd.to_datetime(df['date'])
    sub = df[df['date']<=cut]
    last = sub.iloc[-1]
    print(f"{s:<10}{len(sub):>6}{str(last['date'].date()):>14}{float(last['close']):>15.4f}")
print()
for s in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
    df = pd.read_csv(f"../persistent/index_data/{s}.csv")
    df['date'] = pd.to_datetime(df['date'])
    sub = df[df['date']<=cut]
    print(f"{s:<10}{len(sub):>6}{str(sub.iloc[-1]['date'].date()):>14}{float(sub.iloc[-1]['close']):>15.4f}")