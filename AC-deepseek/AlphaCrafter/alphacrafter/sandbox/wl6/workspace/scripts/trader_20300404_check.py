from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("total_assets:", acct["total_assets"])
print("net_assets:", acct["net_assets"])
print("available_cash:", acct["available_cash"])
print("watch_list:", acct["watch_list"])
print("positions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} w={p['market_value']/acct['net_assets']*100:.2f}%")
print("orders:", len(acct.get("orders", [])))

for s in ["SPX", "VIX", "NDX", "XAU", "WTI", "BTC", "ETH", "US10Y", "000300.SH", "COPPER", "SOX", "N225", "SX5E", "HSI", "000688.SH", "CN10Y"]:
    df = None
    try:
        df = get_stock_daily_data(symbol=s, days=35)
    except Exception:
        df = None
    if df is None or len(df) < 30:
        print(s, "no stock data"); continue
    df = df.sort_values("date")
    r5 = df["close"].iloc[-1]/df["close"].iloc[-6]-1
    r20 = df["close"].iloc[-1]/df["close"].iloc[-21]-1
    print(f"{s}: last={df['close'].iloc[-1]:.2f} r5={r5*100:+.2f}% r20={r20*100:+.2f}%")
