import json, os, glob, time

now = time.time()
print("=== Recently modified files (within 2h) ===")
for root in [".", "../persistent"]:
    for f in glob.glob(root + "/**/*", recursive=True):
        if os.path.isfile(f) and "__pycache__" not in f and not f.endswith(".pyc"):
            mt = os.path.getmtime(f)
            if now - mt < 7200:
                print(round(now - mt, 1), "s ago", f)

print("\n=== account.json summary ===")
acc = json.load(open("../persistent/account.json"))
print("net_assets", acc.get("net_assets"))
print("available_cash", acc.get("available_cash"))
print("orders", acc.get("orders"))
pos = acc.get("positions", [])
print("n_positions", len(pos))
for p in pos:
    print(p.get("symbol"), round(float(p.get("quantity", 0)), 4),
          round(float(p.get("market_value", 0)), 2),
          round(float(p.get("profit_loss_rate", 0)) * 100, 3))
