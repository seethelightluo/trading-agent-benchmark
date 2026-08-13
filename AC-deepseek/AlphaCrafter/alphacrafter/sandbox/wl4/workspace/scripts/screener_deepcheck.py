import pandas as pd, numpy as np

ASOF = "2034-03-20"
assets = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

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

PX = pd.DataFrame(px)
# check for dup dates / weird
print("rows per asset (last 12 dates shown):")
print(PX.tail(12))

RET = PX.pct_change()
print("\ndaily pct change last 8 rows (%), clipped view:")
print((RET.tail(8)*100).round(2).T)

mkt = RET.mean(axis=1)
print("\nmkt daily ret last 10 (%):")
print((mkt.tail(10)*100).round(3))

# look for extreme daily moves
ext = RET[(RET.abs() > 0.5)].stack()
print("\nextreme daily moves >50% (last 20):")
print(ext.tail(20) if len(ext) else "none")

# check frozen assets: last 5 rows
print("\nFrozen check (unique close values in last 60d):")
for a in assets:
    s = PX[a].tail(60)
    print(f"  {a:>9}: n_unique={s.nunique():3d}  last={s.iloc[-1]:.4f}")

# correlation among liquid assets only
liquid = [a for a in assets if PX[a].tail(60).nunique() > 30]
print("\nliquid assets:", liquid)
RETl = RET[liquid]
c60 = RETl.tail(60).corr()
c20 = RETl.tail(20).corr()
v60 = c60.values[np.triu_indices_from(c60.values,1)]
v20 = c20.values[np.triu_indices_from(c20.values,1)]
print(f"avg pairwise corr 60d (liquid): {np.nanmean(v60):.2f} | 20d: {np.nanmean(v20):.2f}")
print(f"corr range 20d: {np.nanmin(v20):.2f}..{np.nanmax(v20):.2f}")
