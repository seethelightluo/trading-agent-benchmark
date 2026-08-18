from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
print("total_assets", round(acc.get("total_assets", 0), 2))
print("net_assets", round(acc.get("net_assets", 0), 2))
print("available_cash", round(acc.get("available_cash", 0), 2))
print("gross_position_rate", acc.get("gross_position_rate"))
print("net_position_rate", acc.get("net_position_rate"))
print("orders:", acc.get("orders"))
print("\npositions:")
for p in acc.get("positions", []):
    print("  %-10s qty=%.4f mv=%.2f plr=%.4f price=%.2f" % (
        p["symbol"], p["quantity"], p["market_value"], p["profit_loss_rate"], p["current_price"]))
print("\nwatch_list:", acc.get("watch_list"))
