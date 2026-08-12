"""Dry-run of next block proposal at 2028-02-24 (decision uses data through prev completed day)."""
import json, math
from pathlib import Path
import strategy as S

DATE_PATH = Path("..") / "persistent" / "date.json"
date_state = json.loads(DATE_PATH.read_text())
print("current_date:", date_state.get("current_date"))
print("visible_through:", date_state.get("visible_through"))
tds = date_state.get("trading_days", [])
print("n trading days:", len(tds), "first:", tds[0], "last:", tds[-1])

from alphacrafter.sim.utils import get_account_dict
acct = get_account_dict()
assets = list(acct.get("watch_list", []))
cur_w = S._current_weights(acct, assets)

ensemble = S._load_ensemble()
print("ensemble factors:", len(ensemble))
built = S.build_target(assets, date_state, ensemble, current_weights=cur_w)
if built is None:
    print("build_target -> None")
else:
    weights, forecast, used, meta = built
    print("used factors:", used)
    print("meta: risk=%.3f vix=%.1f m20=%.4f disp=%.4f n_factors=%d" %
          (meta["risk"], meta["vix"], meta["m20"], meta["disp"], meta["n_factors"]))
    print("cap_map:", meta["cap_map"])
    print("\n%-12s %8s %8s %8s %8s" % ("asset", "cur_w", "tgt_w", "d_w", "fcast10d"))
    turnover = 0.0
    gross_edge = 0.0
    for a in assets:
        cw = cur_w.get(a, 0.0)
        tw = weights[a]
        dw = tw - cw
        turnover += abs(dw)
        fe = forecast[a]
        gross_edge += dw * fe
        flag = ""
        if dw > 0.03: flag = " <== add"
        elif dw < -0.03: flag = " <== cut"
        print("%-12s %8.4f %8.4f %+8.4f %+8.4f%s" % (a, cw, tw, dw, fe, flag))
    print("\nsum tgt:", sum(weights.values()))
    print("one-way turnover: %.4f (%.2f%%)" % (turnover, turnover * 100))
    print("gross edge (sum d_w*fcast): %.4f (%.2fbp)" % (gross_edge, gross_edge * 10000))
    print("migration cost 3bp x turnover: %.2fbp" % (turnover * 3.0))
