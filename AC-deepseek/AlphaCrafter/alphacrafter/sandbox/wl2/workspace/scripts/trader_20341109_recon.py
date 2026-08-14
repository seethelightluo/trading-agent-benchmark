"""Reconstruct the 2034-11-09 proposal (visible through 2034-11-08)."""
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

DECISION = "2034-11-09"
VISIBLE = "2034-11-08"

import alphacrafter.sim.utils as U  # noqa: E402

STOCK_DIR = Path("../persistent/stock_data")
IDX_DIR = Path("../persistent/index_data")
_cache = {}


def _read(symbol, kind):
    key = (symbol, kind)
    if key in _cache:
        return _cache[key]
    p = (STOCK_DIR if kind == "stock" else IDX_DIR) / f"{symbol}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VISIBLE].reset_index(drop=True)
    _cache[key] = df
    return df


def fake_stock(symbol, days=None):
    df = _read(symbol, "stock")
    return df.tail(days).reset_index(drop=True) if days is not None else df


def fake_index(symbol, days=None):
    df = _read(symbol, "index")
    return df.tail(days).reset_index(drop=True) if days is not None else df


U.get_stock_daily_data = fake_stock
U.get_index_daily_data = fake_index

import strategy  # noqa: E402
import importlib  # noqa: E402
importlib.reload(strategy)

date_payload = json.load(open("../persistent/date.json"))
td = date_payload["trading_days"]
date_state = {"current_date": DECISION, "visible_through": VISIBLE, "trading_days": td}
ensemble = strategy._load_ensemble()

# actual current weights at 11-09 (from account)
cur_w = {
    "000300.SH": 67245.67 / 1432194.80, "SPX": 204037.92 / 1432194.80,
    "HSI": 67245.67 / 1432194.80, "N225": 19570.97 / 1432194.80,
    "SX5E": 67245.67 / 1432194.80, "000688.SH": 46040.97 / 1432194.80,
    "SOX": 41697.69 / 1432194.80, "NDX": 201255.06 / 1432194.80,
    "XAU": 207677.52 / 1432194.80, "COPPER": 134289.17 / 1432194.80,
    "WTI": 18251.11 / 1432194.80, "BTC": 67245.67 / 1432194.80,
    "ETH": 32170.80 / 1432194.80, "US10Y": 129110.46 / 1432194.80,
    "CN10Y": 129110.46 / 1432194.80,
}
assets = list(cur_w.keys())
built = strategy.build_target(assets, date_state, ensemble, current_weights=cur_w)
if built is None:
    print("build_target None")
else:
    weights, forecast, used, meta = built
    print("used:", used)
    print("meta risk %.3f vix %.1f m20 %.4f disp %.4f" % (meta["risk"], meta["vix"], meta["m20"], meta["disp"]))
    for a in sorted(weights, key=lambda x: -weights[x]):
        print(f"{a:10s} {weights[a]*100:6.2f}%  r20={meta['r20'][a]*100:7.2f}%  cap={meta['cap_map'].get(a,0.14):.2f}  fcast={forecast[a]*100:6.2f}%")
    print("sum:", sum(weights.values()))
    turn = sum(abs(weights[a] - cur_w[a]) for a in assets)
    print("one-way turnover vs current: %.1f%%" % (turn * 100))
