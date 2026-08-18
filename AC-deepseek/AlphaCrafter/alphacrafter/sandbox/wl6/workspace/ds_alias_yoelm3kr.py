import pandas as pd, os
# Check stock data ranges
for sym in ['SPX','BTC','XAU','CN10Y']:
    p = os.path.join('../persistent/stock_data', f'{sym}.csv')
    df = pd.read_csv(p)
    print(sym, 'rows:', len(df), 'first:', df['date'].iloc[0], 'last:', df['date'].iloc[-1])
# Check VIX range
df = pd.read_csv('../persistent/index_data/VIX.csv')
print('VIX rows:', len(df), 'first:', df['date'].iloc[0], 'last:', df['date'].iloc[-1])
# Check columns of stock data
print(df.columns.tolist())
