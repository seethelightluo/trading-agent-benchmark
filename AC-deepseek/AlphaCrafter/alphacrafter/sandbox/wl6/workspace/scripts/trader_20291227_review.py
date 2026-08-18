from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 2))
print("market_value", round(acc.get("market_value", 0), 2))
print("gross_position_rate", acc.get("gross_position_rate"))
print("net_position_rate", acc.get("net_position_rate"))
print("orders:", len(acc.get("orders", [])))
for o in acc.get("orders", [])[:5]:
    print(" order:", o)
pos = {p["symbol"]: p for p in acc.get("positions", [])}
tot = sum(p["market_value"] for p in pos.values()) or 1.0
for sym in ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]:
    p = pos.get(sym)
    if p:
        print(f"{sym}: qty={p['quantity']:.4f} mv={p['market_value']:.0f} "
              f"w={p['market_value']/tot*100:.2f}% pl={p['profit_loss']:.0f} "
              f"({p['profit_loss_rate']*100:.2f}%)")
    else:
        print(f"{sym}: NO POSITION")
