import pandas as pd, numpy as np, os

DATA = "../persistent/stock_data"
OBS = "../persistent/index_data"
CUT = "2028-11-09"
assets = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load(a, cut=CUT):
    df = pd.read_csv(os.path.join(DATA, a + ".csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'] <= pd.to_datetime(cut)].reset_index(drop=True)
    return df

rows = {}
for a in assets:
    df = load(a)
    close = df['close']
    def r(n):
        return (close.iloc[-1] / close.iloc[-1-n] - 1) * 100 if len(close) > n else np.nan
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    ma20_prev = close.rolling(20).mean().iloc[-6]
    rets = close.pct_change()
    vol20 = rets.tail(20).std() * np.sqrt(252) * 100
    vol60 = rets.tail(60).std() * np.sqrt(252) * 100
    rows[a] = dict(r5=r(5), r10=r(10), r20=r(20), r60=r(60),
                   above_ma20=close.iloc[-1] > ma20, above_ma60=close.iloc[-1] > ma60,
                   ma20_slope_pct=(ma20/ma20_prev-1)*100,
                   vol20=vol20, vol60=vol60,
                   last=close.iloc[-1], last_date=str(df['date'].iloc[-1].date()), n=len(df))

out = pd.DataFrame(rows).T
pd.set_option('display.width', 250)
print(out.round(2).to_string())
print("\nrows per asset (should be similar):", out['n'].astype(int).to_dict())

print("\n=== OBSERVATION SIGNALS (thru 2028-11-09) ===")
for o in ["VIX","DXY","USDCNY","USDJPY","EURUSD"]:
    df = pd.read_csv(os.path.join(OBS, o + ".csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'] <= pd.to_datetime(CUT)].reset_index(drop=True)
    close = df['close']
    def r(n):
        return (close.iloc[-1] / close.iloc[-1-n] - 1) * 100 if len(close) > n else np.nan
    print(f"{o}: last={close.iloc[-1]:.2f} r5={r(5):.2f}% r20={r(20):.2f}% r60={r(60):.2f}% last_date={df['date'].iloc[-1].date()} n={len(df)}")

rets20 = {}; rets60 = {}
for a in assets:
    close = load(a)['close']
    rets20[a] = (close.iloc[-1]/close.iloc[-21]-1)*100
    rets60[a] = (close.iloc[-1]/close.iloc[-61]-1)*100
r20s = pd.Series(rets20); r60s = pd.Series(rets60)
print("\n20d cross-sectional: mean=%.2f%% spread=%.2fpp (max %s %.1f vs min %s %.1f)" %
      (r20s.mean(), r20s.max()-r20s.min(), r20s.idxmax(), r20s.max(), r20s.idxmin(), r20s.min()))
print("60d cross-sectional: mean=%.2f%% spread=%.2fpp (max %s %.1f vs min %s %.1f)" %
      (r60s.mean(), r60s.max()-r60s.min(), r60s.idxmax(), r60s.max(), r60s.idxmin(), r60s.min()))
print("\n# above MA20:", sum(1 for a in assets if rows[a]['above_ma20']), "/15")
print("# above MA60:", sum(1 for a in assets if rows[a]['above_ma60']), "/15")
print("positive 20d:", int((r20s>0).sum()), "/15  positive 60d:", int((r60s>0).sum()), "/15")
print("\n20d sorted:")
print(r20s.sort_values().round(2).to_string())
