"""Trader cycle-32 regime check: account state + 20/60d returns as of 2028-04-06 (data through prev close)."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("total_assets:", round(acct.get("total_assets",0),2), "net:", round(acct.get("net_assets",0),2),
      "cash:", round(acct.get("available_cash",0),2), "gross_pos:", round(acct.get("gross_position_rate",0),4))
print("positions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']:10s} qty={p['quantity']:12.4f} mv={p['market_value']:12.2f} px={p['current_price']:10.4f}")
print("pending orders:", len(acct.get("orders", [])))
for o in acct.get("orders", [])[:5]:
    print("  ", o)

WL = acct.get("watch_list", [])
print("\nwatchlist:", WL)
rows = []
for s in WL:
    df = get_stock_daily_data(symbol=s, days=90)
    if df is None or len(df) < 65:
        rows.append((s, None, None, None)); continue
    df = df.sort_values("date").reset_index(drop=True)
    c = df["close"]
    r20 = c.iloc[-1]/c.iloc[-21]-1 if len(c) > 21 else None
    r60 = c.iloc[-1]/c.iloc[-61]-1 if len(c) > 61 else None
    vol20 = df["pct_change"].tail(20).std()* (252**0.5) if len(df)>20 else None
    rows.append((s, r20, r60, vol20))
print(f"\n{'asset':10s} {'r20':>9s} {'r60':>9s} {'annvol20':>9s}")
for s, r20, r60, v in rows:
    fmt = lambda x: f"{x*100:8.2f}%" if x is not None else "     n/a"
    print(f"{s:10s} {fmt(r20)} {fmt(r60)} {fmt(v)}")

# macro signals
for s in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]:
    df = get_index_daily_data(symbol=s, days=90)
    if df is None or len(df) < 25:
        print(s, "n/a"); continue
    df = df.sort_values("date").reset_index(drop=True)
    c = df["close"]
    r20 = c.iloc[-1]/c.iloc[-21]-1 if len(c) > 21 else None
    print(f"{s:8s} last={c.iloc[-1]:9.3f} r20={r20*100:7.2f}%")
