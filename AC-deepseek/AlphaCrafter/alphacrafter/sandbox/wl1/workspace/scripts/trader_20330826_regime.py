from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd, numpy as np

assets = get_account_dict()["watch_list"]

def snapshot(target):
    closes = {}
    for a in assets:
        df = get_stock_daily_data(symbol=a, days=170)
        if df is None or len(df) == 0:
            closes[a] = None
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        sub = df[df["date"] <= pd.Timestamp(target)]
        if len(sub) < 70:
            closes[a] = None
            continue
        closes[a] = sub.set_index("date")["close"]
    # 20d mean daily ret (cross-asset average)
    rets = []
    for a, s in closes.items():
        if s is not None and len(s) >= 25:
            rets.append(float(s.pct_change().tail(20).mean()))
    m20 = float(np.mean(rets))
    # breadth vs MA20 / MA60
    above20 = 0; above60 = 0; n = 0
    vols = []
    for a, s in closes.items():
        if s is None or len(s) < 65:
            continue
        n += 1
        if float(s.iloc[-1]) > float(s.rolling(20).mean().iloc[-1]):
            above20 += 1
        if float(s.iloc[-1]) > float(s.rolling(60).mean().iloc[-1]):
            above60 += 1
        vols.append(float(s.pct_change().tail(20).std() * np.sqrt(252)))
    print(f"target {target}: 20d mean daily ret {m20*100:.4f}% | breadth MA20 {above20}/{n}, MA60 {above60}/{n} | mean 20d ann vol {np.mean(vols)*100:.1f}%")
    for a, v in sorted(zip([a for a in assets], vols), key=lambda x: -x[1]):
        pass
    volmap = {a: float(s.pct_change().tail(20).std()*np.sqrt(252))*100 for a, s in closes.items() if s is not None and len(s) >= 25}
    top = sorted(volmap.items(), key=lambda x: -x[1])[:5]
    print("   top vol:", [(a, round(v,1)) for a, v in top])
    # 20d / 60d asset returns
    r20 = {}; r60 = {}
    for a, s in closes.items():
        if s is None or len(s) < 130:
            continue
        r20[a] = (float(s.iloc[-1]) / float(s.iloc[-21]) - 1) * 100
        r60[a] = (float(s.iloc[-1]) / float(s.iloc[-61]) - 1) * 100
    print("   20d best:", sorted(r20.items(), key=lambda x: -x[1])[:5])
    print("   20d worst:", sorted(r20.items(), key=lambda x: x[1])[:5])
    print("   60d best:", sorted(r60.items(), key=lambda x: -x[1])[:5])
    print("   60d worst:", sorted(r60.items(), key=lambda x: x[1])[:5])

snapshot("2033-08-11")
print()
snapshot("2033-08-25")

# VIX level
try:
    vix = pd.read_csv("../persistent/index_data/VIX.csv")
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.sort_values("date")
    for t in ["2033-08-11", "2033-08-25"]:
        sub = vix[vix["date"] <= pd.Timestamp(t)]
        if len(sub):
            last = sub.iloc[-1]
            print(f"VIX @ {t}: {float(last['close']):.1f} (date {str(last['date'].date())})")
    # 20d and 60d ago
    v = vix.set_index("date")["close"].astype(float)
    for t in ["2033-08-25"]:
        sub = v[v.index <= pd.Timestamp(t)]
        if len(sub) >= 61:
            print(f"VIX 20d ago: {float(sub.iloc[-21]):.1f}, 60d ago: {float(sub.iloc[-61]):.1f}")
except Exception as e:
    print("vix err", e)
