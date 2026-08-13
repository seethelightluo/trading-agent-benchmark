from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd

acct = get_account_dict()
assets = acct["watch_list"]
# executed weights @0812 from rebalance_history
w0812 = {
 "000300.SH": 0.06584979313243115, "SPX": 0.07058227932680138, "HSI": 0.06842737738108061,
 "N225": 0.08868029966700684, "SX5E": 0.10087953380111066, "000688.SH": 0.05415020686756881,
 "SOX": 0.07963128949690411, "NDX": 0.03303390632190097, "XAU": 0.08796203252170118,
 "COPPER": 0.05851171301722343, "WTI": 0.06148830771195206, "BTC": 0.06457189014793864,
 "ETH": 0.055428116206819825, "US10Y": 0.06597152439126609, "CN10Y": 0.04483173000829428,
}

def close_on(sym, target):
    df = get_stock_daily_data(symbol=sym, days=170)
    if df is None or len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    sub = df[df["date"] <= pd.Timestamp(target)]
    if len(sub) == 0:
        return None
    return float(sub.iloc[-1]["close"]), str(sub.iloc[-1]["date"].date())

print("per-asset block return 0812->0826 (close-to-close):")
tot_contrib = 0.0
rows = []
for a in assets:
    c0 = close_on(a, "2033-08-12")
    c1 = close_on(a, "2033-08-26")
    if c0 is None or c1 is None or c0[0] is None or c1[0] is None:
        print(f"  {a}: MISSING data")
        continue
    r = c1[0] / c0[0] - 1.0
    contrib = w0812[a] * r
    tot_contrib += contrib
    rows.append((a, r, w0812[a], contrib))
rows.sort(key=lambda x: x[3], reverse=True)
for a, r, w, contrib in rows:
    print(f"  {a:10s} ret {r*100:7.2f}%  w {w*100:5.2f}%  contrib {contrib*100:7.2f}pp")
print(f"sum contrib: {tot_contrib*100:.2f}pp")

nav0 = 1185570.5534896806   # pre-trade @0812
nav1 = 1185531.380189571    # post-trade @0812
nav2 = acct["total_assets"] # @0826
print(f"\nNAV: pre-trade 0812 {nav0:.2f} -> post-trade {nav1:.2f} -> 0826 {nav2:.2f}")
print(f"block gross return (post-trade basis): {(nav2/nav1-1)*100:.4f}%")
print(f"trade cost @0812: {nav0-nav1:.2f} on notional 130577.67 (~3bp)")

# also show asset prices to spot flat artifacts
for a in assets:
    c = close_on(a, "2033-08-26")
    print(f"  {a:10s} last close {c}")
