import json
from pathlib import Path

acc = json.loads(Path("../persistent/account.json").read_text())
hist = acc.get("rebalance_history", [])
print("total entries:", len(hist))
for h in hist[-16:]:
    print(h.get("date"), "executed=", h.get("executed"),
          "skip=", h.get("skip_reason", ""), "tow=", round(h.get("one_way_turnover", 0), 3),
          "edge_bp=", round(h.get("gross_edge_bps", 0), 2), "thr=", round(h.get("decision_edge_threshold_bps", 0), 3))
