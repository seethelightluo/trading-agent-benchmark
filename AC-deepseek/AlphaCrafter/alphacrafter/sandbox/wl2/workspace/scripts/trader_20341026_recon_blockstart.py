"""Reconstruct the 2034-10-26 block-start decision (visible through 2034-10-25).

Monkeypatches get_stock_daily_data / get_index_daily_data to serve CSV data
truncated at the decision date, then runs strategy.build_target to recover the
proposed weights, forecast returns and gates that the live rebalance used.
"""
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

DECISION = "2034-10-26"
VISIBLE = "2034-10-25"

import alphacrafter.sim.utils as U  # noqa: E402

STOCK_DIR = Path("../persistent/stock_data")
IDX_DIR = Path("../persistent/index_data")

_cache = {}


def _read(symbol, kind):
    key = (symbol, kind)
    if key in _cache:
        return _cache[key]
    if kind == "stock":
        p = STOCK_DIR / f"{symbol}.csv"
    else:
        p = IDX_DIR / f"{symbol}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VISIBLE].reset_index(drop=True)
    _cache[key] = df
    return df


def fake_stock(symbol, days=None):
    df = _read(symbol, "stock")
    if days is not None:
        df = df.tail(days).reset_index(drop=True)
    return df


def fake_index(symbol, days=None):
    df = _read(symbol, "index")
    if days is not None:
        df = df.tail(days).reset_index(drop=True)
    return df


U.get_stock_daily_data = fake_stock
U.get_index_daily_data = fake_index

import strategy  # noqa: E402

# reload strategy so it picks up the patched utils
import importlib  # noqa: E402

importlib.reload(strategy)

date_payload = json.load(open("../persistent/date.json"))
td = date_payload["trading_days"]
date_state = {"current_date": DECISION, "visible_through": VISIBLE, "trading_days": td}

ensemble = strategy._load_ensemble()
print("ensemble:", [(f["factor_id"], round(f["weight"], 4)) for f in ensemble])

assets = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

# current weights: cycle-132 executed target (10-12) as approximation
cur_w = {
    "SPX": 0.1400, "NDX": 0.1400, "XAU": 0.1400, "US10Y": 0.0892, "CN10Y": 0.0892,
    "COPPER": 0.0763, "N225": 0.0590, "000300.SH": 0.0462, "HSI": 0.0462,
    "SX5E": 0.0462, "BTC": 0.0462, "SOX": 0.0307, "ETH": 0.0210,
    "000688.SH": 0.0171, "WTI": 0.0126,
}

built = strategy.build_target(assets, date_state, ensemble, current_weights=cur_w)
if built is None:
    print("build_target returned None")
else:
    weights, forecast, used, meta = built
    print("used factors:", used)
    print("meta: risk %.3f vix %.1f m20 %.4f disp %.4f" % (meta["risk"], meta["vix"], meta["m20"], meta["disp"]))
    print("--- target weights ---")
    for a in sorted(weights, key=lambda x: -weights[x]):
        print(f"{a:10s} {weights[a]*100:6.2f}%  r20={meta['r20'][a]*100:7.2f}%  cap={meta['cap_map'].get(a,0.14):.2f}  fcast={forecast[a]*100:6.2f}%")
    print("sum:", sum(weights.values()))
    turn = sum(abs(weights[a] - cur_w.get(a, 0.0)) for a in assets)
    print("one-way turnover vs cycle132 target: %.1f%%" % (turn * 100))
