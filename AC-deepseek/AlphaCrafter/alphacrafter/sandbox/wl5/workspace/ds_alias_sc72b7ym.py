import pandas as pd
df = pd.read_csv('../persistent/stock_data/SPX.csv')
df['date'] = pd.to_datetime(df['date'])
sub = df[df['date'] <= '2029-11-01']
print("last trading day <= 2029-11-01:", sub['date'].iloc[-1])
sub2 = df[df['date'] <= '2029-10-31']
print("last trading day <= 2029-10-31:", sub2['date'].iloc[-1])
print(sub2['date'].iloc[-5:].tolist())
