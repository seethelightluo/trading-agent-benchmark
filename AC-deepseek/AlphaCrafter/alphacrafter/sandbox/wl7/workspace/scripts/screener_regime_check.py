import pandas as pd, numpy as np

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
obs = ['DXY','USDCNY','USDJPY','EURUSD','VIX']

def load(sym, path='../persistent/stock_data'):
    df = pd.read_csv(f'{path}/{sym}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df.iloc[:,0])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df['close'].astype(float)

print("=== TRADABLE ASSETS (through 2027-02-01) ===")
print(f"{'sym':10s} {'last':>12s} {'r5d%':>8s} {'r20d%':>8s} {'r40d%':>8s} {'r60d%':>8s} {'ma20pos':>8s} {'ma60pos':>8s} {'vol20%':>7s} {'maxdd60%':>9s}")
px = {}
for a in assets:
    try:
        s = load(a)
        s = s[s.index <= '2027-02-01']
        px[a] = s
        last = s.iloc[-1]
        r5 = s.iloc[-1]/s.iloc[-6]-1 if len(s)>6 else np.nan
        r20 = s.iloc[-1]/s.iloc[-21]-1 if len(s)>21 else np.nan
        r40 = s.iloc[-1]/s.iloc[-41]-1 if len(s)>41 else np.nan
        r60 = s.iloc[-1]/s.iloc[-61]-1 if len(s)>61 else np.nan
        ma20 = s.rolling(20).mean().iloc[-1]
        ma60 = s.rolling(60).mean().iloc[-1]
        vol20 = s.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252)*100
        dd = (s/s.cummax()-1).min()*100
        print(f"{a:10s} {last:12.2f} {r5*100:8.2f} {r20*100:8.2f} {r40*100:8.2f} {r60*100:8.2f} {last/ma20:8.3f} {last/ma60:8.3f} {vol20:7.1f} {dd:9.1f}")
    except Exception as e:
        print(f"{a:10s} ERROR {e}")

print("\n=== OBSERVATION MACRO ===")
for a in obs:
    try:
        s = load(a, '../persistent/index_data')
        s = s[s.index <= '2027-02-01']
        last = s.iloc[-1]
        r5 = s.iloc[-1]/s.iloc[-6]-1 if len(s)>6 else np.nan
        r20 = s.iloc[-1]/s.iloc[-21]-1 if len(s)>21 else np.nan
        print(f"{a:10s} last={last:10.2f} r5d={r5*100:7.2f}% r20d={r20*100:7.2f}%")
    except Exception as e:
        print(f"{a:10s} ERROR {e}")

# Cross-sectional dispersion & correlation of the 15 tradable assets
rets = pd.DataFrame(px).pct_change().dropna()
rets = rets[rets.index <= '2027-02-01']
disp20 = rets.tail(20).std(axis=1).mean()*100
disp60 = rets.tail(60).std(axis=1).mean()*100
corr20 = rets.tail(20).corr().abs().values[np.triu_indices(15,1)].mean()
corr60 = rets.tail(60).corr().abs().values[np.triu_indices(15,1)].mean()
print(f"\n=== CROSS-SECTION ===")
print(f"dispersion/day 20d={disp20:.3f}% 60d={disp60:.3f}%")
print(f"mean |corr| 20d={corr20:.3f} 60d={corr60:.3f}")
print(f"avg 20d ret of 15 assets: {rets.tail(20).mean().mean()*100:.2f}%")
print(f"median 20d ret: {rets.tail(20).mean().median()*100:.2f}%")
print(f"20d ret by asset:")
for a in assets:
    if a in rets:
        print(f"  {a:10s} {rets.tail(20).mean()[a]*100:7.2f}%")
