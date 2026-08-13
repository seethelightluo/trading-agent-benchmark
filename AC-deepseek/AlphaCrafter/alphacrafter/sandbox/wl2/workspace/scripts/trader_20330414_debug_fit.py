"""Debug: check _fit_weights behavior with WTI cap."""
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from alphacrafter.sim.utils import get_account_dict
import strategy as S

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
date_state = json.loads(Path("../persistent/date.json").read_text())
cur_w = S._current_weights(acc, assets)
ensemble = S._load_ensemble()
w, fc, used, meta = S.build_target(assets, date_state, ensemble, current_weights=cur_w)

# replicate r20 inside build_target
closes = S._closes(assets)
r20 = {a: float(closes[a].iloc[-1] / closes[a].iloc[-21] - 1.0) for a in assets if closes.get(a) is not None and len(closes[a]) >= 21}
print("WTI r20:", r20.get("WTI"))
print("cap_map WTI:", meta["cap_map"].get("WTI"))

# directly test _fit_weights with a cap for WTI
pref = {a: 0.066 for a in assets}
cap_map = {"WTI": 0.09}
fw = S._fit_weights(pref, cap=0.14, floor=0.012, cap_map=cap_map)
print("WTI fitted weight with cap 0.09:", fw["WTI"], "sum:", sum(fw.values()))

# check WTI close data
df = __import__("alphacrafter.sim.utils", fromlist=["get_stock_daily_data"]).get_stock_daily_data("WTI", days=30)
print(df[["date", "close"]].tail(5).to_string())
print("close[-1]:", df["close"].iloc[-1], "close[-21]:", df["close"].iloc[-21])
