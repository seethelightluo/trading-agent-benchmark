import pandas as pd, numpy as np, json, glob, os

ASOF = "2034-03-20"
assets = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
macro  = ["VIX","DXY","USDCNY","USDJPY","EURUSD"]

def load(p):
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df

px = {}
for a in assets:
    df = load(f"../persistent/stock_data/{a}.csv")
    df = df[df.index <= ASOF]
    px[a] = df["close"].astype(float)

mpx = {}
for m in macro:
    df = load(f"../persistent/index_data/{m}.csv")
    df = df[df.index <= ASOF]
    mpx[m] = df["close"].astype(float)

print("Last data date:", max(d.index.max() for d in px.values()).date())

PX = pd.DataFrame(px)
RET = PX.pct_change()

# --- market level (equal-weight cross-asset) ---
mkt = RET.mean(axis=1)
last = PX.index[-1]

def ret_over(df, n):
    return df.iloc[-1] / df.iloc[-1-n] - 1 if len(df) > n else np.nan

print("\n=== Cross-asset market (equal-weight of 15) ===")
print(f"Last date: {last.date()}")
for n,lab in [(5,"5d"),(10,"10d"),(21,"1m"),(63,"3m"),(126,"6m"),(252,"1y")]:
    if len(mkt) > n:
        print(f"  mkt {lab:>4}: {ret_over(mkt,n)*100:+.2f}%")

# trend: 20d vs 60d MA of mkt price
mkt_px = (1+mkt).cumprod()
ma20 = mkt_px.rolling(20).mean(); ma60 = mkt_px.rolling(60).mean()
print(f"  mkt px vs MA20: {'above' if mkt_px.iloc[-1]>ma20.iloc[-1] else 'below'} ({(mkt_px.iloc[-1]/ma20.iloc[-1]-1)*100:+.2f}%)")
print(f"  mkt px vs MA60: {'above' if mkt_px.iloc[-1]>ma60.iloc[-1] else 'below'} ({(mkt_px.iloc[-1]/ma60.iloc[-1]-1)*100:+.2f}%)")
print(f"  MA20 vs MA60:   {'bull' if ma20.iloc[-1]>ma60.iloc[-1] else 'bear'} slope")

# realized vol
rv20 = mkt.tail(20).std()*np.sqrt(252)
rv60 = mkt.tail(60).std()*np.sqrt(252)
print(f"  mkt realized vol 20d: {rv20*100:.1f}% ann | 60d: {rv60*100:.1f}% ann")

# drawdown from 252d high
peak = mkt_px.tail(252).max()
print(f"  mkt drawdown from 1y high: {(mkt_px.iloc[-1]/peak-1)*100:+.2f}%")

# consecutive direction days
signs = np.sign(mkt.tail(20))
run = 1
for i in range(len(signs)-2, -1, -1):
    if signs.iloc[i] == signs.iloc[-1]: run += 1
    else: break
print(f"  last {run} consecutive {'up' if signs.iloc[-1]>0 else 'down'} days")

print("\n=== Per-asset 10d/21d/63d returns (%) ===")
out = {}
for a in assets:
    s = PX[a]
    r = {lab: ret_over(s,n)*100 for n,lab in [(10,"r10"),(21,"r21"),(63,"r63")]}
    v = s.pct_change().tail(20).std()*np.sqrt(252)*100
    out[a] = {**r, "vol20": v}
    print(f"  {a:>9}: r10 {r['r10']:+7.2f}  r21 {r['r21']:+7.2f}  r63 {r['r63']:+8.2f}  vol20 {v:5.1f}%")

print("\n=== Macro (last values & 20d change) ===")
for m in macro:
    s = mpx[m]
    chg = (s.iloc[-1]/s.iloc[-21]-1)*100 if len(s)>21 else np.nan
    print(f"  {m:>7}: {s.iloc[-1]:.2f}  20d chg {chg:+.2f}%")

# correlation regime: avg pairwise 60d corr of daily returns
c60 = RET.tail(60).corr()
avg_corr = (c60.values[np.triu_indices_from(c60.values,1)]).mean()
c20 = RET.tail(20).corr()
avg_corr20 = (c20.values[np.triu_indices_from(c20.values,1)]).mean()
print(f"\nAvg pairwise corr (60d): {avg_corr:.2f} | (20d): {avg_corr20:.2f}")

# dispersion
disp = RET.tail(20).std(axis=1).mean()*100
print(f"Cross-sectional dispersion (20d, daily std of asset rets): {disp:.2f}%")

# momentum dispersion: spread between top/bottom 21d return
r21 = PX.iloc[-1]/PX.iloc[-22]-1
print(f"21d return spread max-min: {(r21.max()-r21.min())*100:.1f}% | top: {r21.idxmax()} {r21.max()*100:+.1f}% | bot: {r21.idxmin()} {r21.min()*100:+.1f}%")

# trend-following regime proxy: momentum IC (rank corr of 21d ret vs next? can't; just report 60d momentum sign)
mom60 = PX.iloc[-1]/PX.iloc[-61]-1
print("\n60d momentum per asset:")
for a in assets:
    print(f"  {a:>9}: {mom60[a]*100:+7.2f}%")

# save a snapshot for later
snap = {"asof": ASOF, "last": str(last.date()),
        "mkt_r10": ret_over(mkt,10)*100, "mkt_r21": ret_over(mkt,21)*100,
        "mkt_r63": ret_over(mkt,63)*100, "rv20": rv20*100, "rv60": rv60*100,
        "dd_1y": (mkt_px.iloc[-1]/peak-1)*100, "avg_corr60": avg_corr,
        "avg_corr20": avg_corr20, "disp20": disp}
with open("scripts/regime_snapshot.json","w") as f:
    json.dump(snap, f, indent=2)
print("\nsaved scripts/regime_snapshot.json")
