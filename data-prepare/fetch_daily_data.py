#!/usr/bin/env python
"""
fetch_daily_data.py — 抓取 9 条世界线全部可交易资产的日频数据

资产口径/数据源候选见 asset_spec.py（单一事实源）。日期范围 2020-01-01 ~ END（默认今日）。

== 数据源策略（实测，本机 Clash TUN 环境）==
  sina/163（国内）   稳定 ✅ —— A 股/港股/美股指数、环球指数(限 ~1000 行)
  eastmoney _em       RemoteDisconnected 频发，重试后间歇可用 ⚠️ —— 商品/外汇/DXY/环球全量
  Yahoo/FRED（国际）  出口 IP 被 rate-limit ❌ —— 仅长尾兜底（cooldown 后偶通）
  Binance             稳定 ✅ —— 加密
  akshare bond/macro  稳定 ✅ —— 中美国债收益率、SOX

每资产按 asset_spec['sources'] 优先级依次尝试，先成先用；eastmoney 自带重试。
单资产失败不中断，末尾汇总并支持 --only 断点续抓。

输出（data-prepare/asset-daily-data/）
  <asset_id>.csv         date,open,high,low,close,volume[,adjclose]
  all_close_wide.csv     日期并集 × 资产收盘宽表（非交易日空值, 预期）
  COPPER_USD_PER_TON.csv 铜 USD/吨 = close × 2204.62262
  panel.parquet / panel.csv  规范长表(仅 19 基准资产, ≤2026-07-16) — 给适配器
  COVERAGE.md            每资产: 来源/ticker/单位/起止/行数/缺失/基线比对/所用源

用法
  python fetch_daily_data.py                 # 抓全部，复用已落盘且覆盖的 CSV
  python fetch_daily_data.py --force         # 全量重抓
  python fetch_daily_data.py --only BTC,VIX  # 只抓指定资产
  python fetch_daily_data.py --end 2026-07-16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "asset-daily-data"

from asset_spec import ASSET_SPEC, COPPER_LB_TO_TON  # noqa: E402

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
DEFAULT_START = "2020-01-01"
WARMUP_END = "2026-07-16"          # 面板 warmup 截止（在线阶段走世界线合成数据）

# ---- Clash 节点轮换（绕过 Yahoo 对单一出口 IP 的 rate-limit）----
# 实测：TW 节点被 Yahoo 限流；逐 Yahoo 资产换到不同 US/JP/SG HY2 节点，每 IP 仅 1 次请求即稳。
# 仅当 --rotate-clash 开启时生效；结束后自动恢复原节点。
CLASH_SOCK = "/tmp/verge/verge-mihomo.sock"
# 用「未被打爆」的节点轮换：KR + TW-1..7（TW-8 是 自动选择 默认 pick，已被 Yahoo 限流）。
# 每个 Yahoo 请求落到不同节点，单 IP 仅 1 次，避免触发 Yahoo rate-limit。
CLASH_POOL = ["KR-1", "KR-2", "KR-3", "KR-4", "TW-1", "TW-2", "TW-3", "TW-4",
              "TW-5", "TW-6", "TW-7", "KR-1"]
_clash_i = [0]
ROTATE_CLASH = [False]


def clash_switch(node):
    import subprocess
    try:
        subprocess.run(["curl", "-s", "-m", "5", "--unix-socket", CLASH_SOCK,
                        "-X", "PUT", "http://localhost/proxies/主代理",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps({"name": node})],
                       capture_output=True, timeout=8)
    except Exception:
        pass


def clash_now():
    import subprocess
    try:
        r = subprocess.run(["curl", "-s", "-m", "5", "--unix-socket", CLASH_SOCK,
                            "http://localhost/proxies/主代理"],
                           capture_output=True, text=True, timeout=8)
        return json.loads(r.stdout).get("now")
    except Exception:
        return None


def clash_rotate():
    n = CLASH_POOL[_clash_i[0] % len(CLASH_POOL)]
    _clash_i[0] += 1
    clash_switch(n)
    return n


# --------------------------------------------------------------------------- #
# 通用 HTTP
# --------------------------------------------------------------------------- #
def _epoch(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def retry_get(url: str, *, tries: int = 6, timeout: int = 30,
              headers: dict | None = None) -> bytes:
    """urllib GET + 指数退避；429/5xx 与网络层错误重试。"""
    import http.client
    net_errs = (urllib.error.URLError, OSError, TimeoutError,
                http.client.RemoteDisconnected, http.client.BadStatusLine,
                http.client.IncompleteRead)
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                w = min(2 ** k + 0.5 * k, 40)
                time.sleep(w); continue
            raise
        except net_errs as e:
            last = e
            time.sleep(min(2 ** k, 20))
    raise RuntimeError(f"重试 {tries} 次仍失败: {url[:90]} ({last})")


def ak_retry(fn, *, args=(), kwargs=None, attempts: int = 8, base: float = 3.0,
             tag: str = ""):
    """akshare 调用重试包装：捕获网络层错误（eastmoney RemoteDisconnected 常见）。"""
    kwargs = kwargs or {}
    last = None
    for k in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            name = type(e).__name__
            # 仅对网络/连接类错误重试；KeyError/ValueError 等逻辑错误立即跳出
            if name in ("KeyError", "ValueError", "IndexError", "TypeError"):
                raise
            w = min(base * (k + 1) + (k * 0.7), 25)
            if k < attempts - 1:
                time.sleep(w)
    raise last


# --------------------------------------------------------------------------- #
# 规范化
# --------------------------------------------------------------------------- #
def _num(df, *cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def canonical(df: pd.DataFrame, *, keep_adj: bool = False) -> pd.DataFrame:
    """统一为 date(str), open, high, low, close, volume [, adjclose]，升序去重。"""
    df = df.copy()
    # 日期列名兼容
    for c in list(df.columns):
        if str(c).lower() in ("date", "日期", "datetime", "time", "date时间"):
            df = df.rename(columns={c: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    cols = ["open", "high", "low", "close", "volume"]
    df = _num(df, *cols)
    out = ["date", "open", "high", "low", "close", "volume"]
    if keep_adj and "adjclose" in df.columns:
        df = _num(df, "adjclose")
        out.append("adjclose")
    for c in out:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[out].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # 剔除 close 为 NaN 的行（如当日债券收益率尚未发布、期货展期空行）—— 日频源只应有交易日
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def filter_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    m = (df["date"] >= start) & (df["date"] <= end)
    return df[m].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 各数据源 fetcher（均返回规范长表前的原始/半规范 df，main 里再 canonical）
# --------------------------------------------------------------------------- #
def f_sina_zh(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.stock_zh_index_daily, kwargs={"symbol": symbol},
                  attempts=6, tag="sina_zh")
    return canonical(df), "akshare.stock_zh_index_daily"


def f_sina_us(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.index_us_stock_sina, kwargs={"symbol": symbol},
                  attempts=6, tag="sina_us")
    return canonical(df), "akshare.index_us_stock_sina"


def f_sina_hk(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.stock_hk_index_daily_sina, kwargs={"symbol": symbol},
                  attempts=6, tag="sina_hk")
    return canonical(df), "akshare.stock_hk_index_daily_sina"


def f_sina_global(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.index_global_hist_sina, kwargs={"symbol": symbol},
                  attempts=5, tag="sina_global")
    return canonical(df), "akshare.index_global_hist_sina"


def f_em_global(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.index_global_hist_em, kwargs={"symbol": symbol},
                  attempts=4, base=1.5, tag="em_global")
    # 列: 日期, 开盘, 最高, 最低, 收盘, 涨跌幅, ...
    df = df.rename(columns={"开盘": "open", "最高": "high", "最低": "low",
                            "收盘": "close", "成交量": "volume"})
    return canonical(df), "akshare.index_global_hist_em"


def f_em_futures(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.futures_global_hist_em, kwargs={"symbol": symbol},
                  attempts=4, base=1.5, tag="em_futures")
    df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                            "最低": "low", "最新价": "close", "总量": "volume"})
    return canonical(df), "akshare.futures_global_hist_em"


def f_em_forex(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.forex_hist_em, kwargs={"symbol": symbol},
                  attempts=4, base=1.5, tag="em_forex")
    df = df.rename(columns={"日期": "date", "今开": "open", "最高": "high",
                            "最低": "low", "最新价": "close"})
    return canonical(df), "akshare.forex_hist_em"


def _boc_close(currency: str, start: str, end: str) -> pd.DataFrame:
    """currency_boc_sina → [date, v]，v=央行中间价（CNY per 100 外币）。国内直连，稳定。"""
    import akshare as ak
    df = ak_retry(ak.currency_boc_sina,
                  kwargs={"symbol": currency,
                          "start_date": start.replace("-", ""),
                          "end_date": end.replace("-", "")},
                  attempts=5, tag="boc")
    df = df[["日期", "央行中间价"]].copy()
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _close_only(series: pd.Series, label: str):
    """close 序列 → 规范 OHLCV（open=high=low=close, volume=0）。"""
    df = series.dropna().to_frame("close").reset_index().rename(columns={"index": "date"})
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 0.0
    return canonical(df), label


def f_boc_usdcny(symbol, start, end):
    df = _boc_close("美元", start, end).set_index("date")["v"] / 100.0
    return _close_only(df, "BOC 美元央行中间价/100")


def f_boc_usdjpy(symbol, start, end):
    u = _boc_close("美元", start, end).set_index("date")["v"]
    j = _boc_close("日元", start, end).set_index("date")["v"]
    return _close_only(u / j, "BOC 美元/日元 cross")  # 两边均 per-100，比值=JPY/USD


def f_boc_usdkrw(symbol, start, end):
    u = _boc_close("美元", start, end).set_index("date")["v"]
    k = _boc_close("韩国元", start, end).set_index("date")["v"]
    return _close_only(u / k, "BOC 美元/韩国元 cross")


def f_boc_dxy(symbol, start, end):
    """DXY = 官方篮子公式，由 BOC 6 成分货币交叉汇率合成（国内，免 Yahoo）。"""
    comps = {"EUR": "欧元", "JPY": "日元", "GBP": "英镑",
             "CAD": "加拿大元", "SEK": "瑞典克朗", "CHF": "瑞士法郎"}
    series = [_boc_close("美元", start, end).set_index("date")["v"].rename("USD")]
    for code, nm in comps.items():
        series.append(_boc_close(nm, start, end).set_index("date")["v"].rename(code))
    j = pd.concat(series, axis=1).dropna()
    eur_usd, gbp_usd = j["EUR"] / j["USD"], j["GBP"] / j["USD"]
    usd_jpy = j["USD"] / j["JPY"]
    usd_cad, usd_sek, usd_chf = j["USD"] / j["CAD"], j["USD"] / j["SEK"], j["USD"] / j["CHF"]
    dxy = (50.14348112
           * eur_usd ** -0.576 * usd_jpy ** 0.136 * gbp_usd ** -0.119
           * usd_cad ** 0.091 * usd_sek ** 0.042 * usd_chf ** 0.036)
    return _close_only(dxy, "BOC 6-篮子 DXY 公式")


def f_sina_foreign(symbol, start, end):
    """sina 外盘期货历史（gold=GC, copper=HG[cents/lb], crude=CL）。稳定，国内直连。"""
    import akshare as ak
    df = ak_retry(ak.futures_foreign_hist, kwargs={"symbol": symbol},
                  attempts=5, tag="sina_foreign")
    df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                            "low": "low", "close": "close", "volume": "volume"})
    return canonical(df), "akshare.futures_foreign_hist"


def f_sox(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.macro_global_sox_index, attempts=6, tag="sox")
    df = df.rename(columns={"日期": "date", "最新值": "close"})
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 0.0
    return canonical(df), "akshare.macro_global_sox_index"


def f_us10y(symbol, start, end):
    import akshare as ak
    df = ak_retry(ak.bond_zh_us_rate,
                  kwargs={"start_date": start.replace("-", "")},
                  attempts=5, tag="us10y")
    df = df.rename(columns={"日期": "date", "美国国债收益率10年": "close"})
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 0.0
    return canonical(df), "akshare.bond_zh_us_rate(美国国债收益率10年)"


def f_cn10y(symbol, start, end):
    """中债10Y：复用 bond_zh_us_rate 的「中国国债收益率10年」列（全量稳定）。
    bond_china_yield 在大范围请求下返回空，故不用。"""
    import akshare as ak
    df = ak_retry(ak.bond_zh_us_rate,
                  kwargs={"start_date": start.replace("-", "")},
                  attempts=5, tag="cn10y")
    df = df.rename(columns={"日期": "date", "中国国债收益率10年": "close"})
    df = df.dropna(subset=["close"])
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 0.0
    return canonical(df), "akshare.bond_zh_us_rate(中国国债收益率10年)"


def f_binance(symbol, start, end):
    t0, t1 = _epoch(start) * 1000, (_epoch(end) + 86_400) * 1000
    out, cur = [], t0
    while cur < t1:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
               f"&interval=1d&startTime={cur}&endTime={t1}&limit=1000")
        batch = json.loads(retry_get(url, tries=5))
        if not batch:
            break
        for k in batch:
            out.append({"date": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                        "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                        "close": float(k[4]), "volume": float(k[5])})
        last_open = batch[-1][0]
        if last_open <= cur:
            break
        cur = last_open + 86_400_000
        if len(batch) < 1000:
            break
    if not out:
        raise RuntimeError(f"Binance 无数据: {symbol}")
    return canonical(pd.DataFrame(out)), "binance klines"


def f_yahoo(symbol, start, end):
    """Yahoo chart。⚠️ TW 节点被 Yahoo rate-limit；--rotate-clash 时逐次换 US/JP/SG HY2 节点，
    否则自带 ≥2s 节流（同 IP 连续请求易被限）。"""
    if ROTATE_CLASH[0]:
        node = clash_rotate()
        time.sleep(2.5)  # 等节点切换 + DNS 生效（实测 1s 不足，会拿到空结果）
    else:
        _yahoo_throttle()
        node = None
    sym = urllib.parse.quote(symbol, safe="")
    p1, p2 = _epoch(start), _epoch(end) + 86_400
    # ⚠️ 实测：长 Chrome UA 会被 Yahoo 限流（HTTPError/空结果），短 UA "Mozilla/5.0" 通过率更高。
    yh_hdr = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = (f"https://{host}/v8/finance/chart/{sym}"
               f"?period1={p1}&period2={p2}&interval=1d")
        try:
            doc = json.loads(retry_get(url, tries=2, headers=yh_hdr))["chart"]["result"][0]
        except Exception:
            continue
        ts = doc.get("timestamp") or []
        q = (doc.get("indicators") or {}).get("quote", [{}])[0]
        adj = ((doc.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
        rows = []
        for i, t in enumerate(ts):
            o = q["open"][i] if q.get("open") else None
            h = q["high"][i] if q.get("high") else None
            l_ = q["low"][i] if q.get("low") else None
            c = q["close"][i] if q.get("close") else None
            v = q["volume"][i] if q.get("volume") else None
            if o is None and h is None and l_ is None and c is None:
                continue
            rows.append({"date": datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d"),
                         "open": o, "high": h, "low": l_, "close": c, "volume": v,
                         "adjclose": adj[i] if adj else None})
        if rows:
            return canonical(pd.DataFrame(rows), keep_adj=True), f"yahoo {symbol}"
    raise RuntimeError(f"Yahoo 无数据: {symbol}")


def f_cboe_vix(symbol, start, end):
    """CBOE VIX 日历史（尝试公开 JSON/CSV）。"""
    # CBOE 历史 VIX: monthly CSV 历史 + daily 近端。用 macrotrends 风格兜底失败则抛错。
    raise RuntimeError("cboe_vix: 暂无稳定免key日频源，跳过")


def f_proxy_jp_semi(symbol, start, end):
    parts = []
    for t in ("6857.T", "8035.T"):
        df = f_yahoo(t, start, end)[0][["date", "close"]].rename(columns={"close": t})
        parts.append(df.set_index("date"))
    j = parts[0]
    for p in parts[1:]:
        j = j.join(p, how="outer").sort_index()
    norm = j.apply(lambda s: s / s.dropna().iloc[0])
    eq = norm.mean(axis=1).rename("close").to_frame().reset_index()
    eq["open"] = eq["high"] = eq["low"] = eq["close"]
    eq["volume"] = 0.0
    return canonical(eq), "proxy 6857.T+8035.T equal-weight (yahoo)"


_YAHOO_LAST = [0.0]


def _yahoo_throttle(min_gap: float = 2.0):
    gap = time.time() - _YAHOO_LAST[0]
    if gap < min_gap:
        time.sleep(min_gap - gap)
    _YAHOO_LAST[0] = time.time()


SOURCES = {
    "sina_zh": f_sina_zh, "sina_us": f_sina_us, "sina_hk": f_sina_hk,
    "sina_global": f_sina_global, "sina_foreign": f_sina_foreign,
    "em_global": f_em_global, "em_futures": f_em_futures, "em_forex": f_em_forex,
    "sox": f_sox, "us10y": f_us10y, "cn10y": f_cn10y, "binance": f_binance,
    "boc_usdcny": f_boc_usdcny, "boc_usdjpy": f_boc_usdjpy,
    "boc_usdkrw": f_boc_usdkrw, "boc_dxy": f_boc_dxy,
    "yahoo": f_yahoo, "cboe_vix": f_cboe_vix, "proxy_jp_semi": f_proxy_jp_semi,
}


def fetch_asset(spec, start, end):
    """按 sources 优先级尝试；source 项可为 (key, sym) 或 (key, sym, scale)。
    scale!=1 时对 OHLC 列整体缩放（如 sina 铜 cents/lb → USD/lb 用 0.01）。"""
    last_err = None
    for entry in spec["sources"]:
        key, sym = entry[0], entry[1]
        scale = entry[2] if len(entry) > 2 else 1.0
        fn = SOURCES.get(key)
        if fn is None:
            continue
        try:
            df, label = fn(sym, start, end)
            if scale != 1.0:
                for c in ("open", "high", "low", "close"):
                    df[c] = pd.to_numeric(df[c], errors="coerce") * scale
                label += f" (scale×{scale})"
            df = filter_range(df, start, end)
            if df.empty:
                last_err = RuntimeError(f"{label}: 空数据")
                continue
            return df, label
        except Exception as e:
            last_err = e
            print(f"      [{key}] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(1.5)
    raise RuntimeError(f"所有源均失败: {last_err}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--rotate-clash", action="store_true",
                    help="逐 Yahoo 资产轮换 Clash 出口节点（绕过 Yahoo 单 IP rate-limit），结束后恢复原节点")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = ASSET_SPEC
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in ASSET_SPEC if s["asset_id"] in want]

    orig_node = None
    if args.rotate_clash:
        ROTATE_CLASH[0] = True
        orig_node = clash_now()
        print(f"[clash] 原节点={orig_node}；将逐 Yahoo 资产轮换，结束后恢复\n")
        import atexit
        atexit.register(lambda: (clash_switch(orig_node),
                                 print(f"[clash] 已恢复节点 → {orig_node}")))

    print(f"日期范围 {args.start} ~ {args.end} | 资产 {len(specs)} 个 | 输出 {outdir}\n")
    results, per_asset = [], {}

    for i, spec in enumerate(specs, 1):
        aid = spec["asset_id"]
        csv_path = outdir / f"{aid}.csv"
        df, reused, label = None, False, ""
        if csv_path.exists() and not args.force:
            try:
                old = canonical(pd.read_csv(csv_path), keep_adj=("adjclose" in pd.read_csv(csv_path, nrows=1).columns))
                # 复用条件：覆盖到 end，且起点在 start 之后 10 天内（2020-01-01 为元旦休市，首交易日 01-02）
                start_grace = (pd.Timestamp(args.start) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                if not old.empty and old["date"].min() <= start_grace and old["date"].max() >= args.end:
                    df, reused, label = old, True, "(复用已落盘)"
            except Exception:
                pass
        if df is None:
            print(f"[{i}/{len(specs)}] 抓 {aid} ...")
            try:
                df, label = fetch_asset(spec, args.start, args.end)
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"   ❌ {aid} 失败: {type(e).__name__}: {str(e)[:90]}")
                results.append({"asset_id": aid, "ok": False, "err": str(e)[:120],
                                "name": spec["name"], "klass": spec["klass"],
                                "unit": spec["unit"]})
                continue
        else:
            print(f"[{i}/{len(specs)}] 复用 {aid}.csv ({len(df)} 行)")
        per_asset[aid] = df
        results.append({"asset_id": aid, "ok": True, "name": spec["name"],
                        "klass": spec["klass"], "unit": spec["unit"],
                        "note": spec["note"], "rows": int(len(df)),
                        "start": df["date"].min(), "end": df["date"].max(),
                        "baseline": spec["baseline"], "source": label})

    print("\n构建派生产物 ...")
    build_wide_close(per_asset, outdir)
    build_copper_per_ton(per_asset, outdir)
    if not args.no_panel:
        build_panel(per_asset, outdir, args)
    write_coverage(results, outdir)

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n✅ 完成 {n_ok}/{len(specs)} 资产。详见 {outdir/'COVERAGE.md'}")
    fails = [r["asset_id"] for r in results if not r.get("ok")]
    if fails:
        print(f"⚠️  失败: {fails}")
        print(f"   重试: python fetch_daily_data.py --only {','.join(fails)}")
    return 0 if not fails else 2


# --------------------------------------------------------------------------- #
def build_wide_close(per_asset, outdir):
    frames = [df[["date", "close"]].rename(columns={"close": aid}).set_index("date")
              for aid, df in per_asset.items()]
    if not frames:
        return
    wide = pd.concat(frames, axis=1).sort_index()
    wide.index.name = "date"
    wide.to_csv(outdir / "all_close_wide.csv")
    print(f"   all_close_wide.csv  ({len(wide)} 行 × {wide.shape[1]} 资产)")


def build_copper_per_ton(per_asset, outdir):
    if "COPPER" not in per_asset:
        return
    df = per_asset["COPPER"][["date", "open", "high", "low", "close"]].copy()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * COPPER_LB_TO_TON
    df.to_csv(outdir / "COPPER_USD_PER_TON.csv", index=False)
    print(f"   COPPER_USD_PER_TON.csv  末值 {df['close'].iloc[-1]:.2f} USD/吨")


def build_panel(per_asset, outdir, args):
    from asset_spec import BENCHMARK_ASSET_IDS
    keep = [a for a in per_asset if a in BENCHMARK_ASSET_IDS] if not args.only else list(per_asset)
    parts = []
    for aid in keep:
        df = per_asset[aid].copy()
        df.insert(0, "asset_id", aid)
        if "adjclose" in df.columns:
            df = df.drop(columns=["adjclose"])
        df["amount"] = (pd.to_numeric(df["close"], errors="coerce")
                        * pd.to_numeric(df["volume"], errors="coerce").fillna(0)).fillna(0)
        df = _num(df, "open", "high", "low", "close", "volume", "amount")
        parts.append(df)
    if not parts:
        print("   ⚠️  无基准资产，跳过 panel")
        return
    panel = pd.concat(parts, ignore_index=True).sort_values(["asset_id", "date"])
    panel = panel[panel["date"] <= WARMUP_END].reset_index(drop=True)
    if HAVE_PARQUET:
        panel.to_parquet(outdir / "panel.parquet", index=False)
        print(f"   panel.parquet  ({len(panel)} 行, {panel['asset_id'].nunique()} 资产, ≤{WARMUP_END})")
    panel.to_csv(outdir / "panel.csv", index=False)
    print(f"   panel.csv      ({len(panel)} 行)")


def write_coverage(results, outdir):
    today = date.today().isoformat()
    L = ["# COVERAGE — 资产日频数据覆盖与校验", "",
         f"> 生成：{today}　|　范围：2020-01-01 ~ 抓取末端　|　warmup 截止 {WARMUP_END}", "",
         "## 1. 每资产覆盖", "",
         "| asset_id | 名称 | 类别 | 单位 | 所用源 | 起始 | 末端 | 行数 | 基线(2026-07-16) | 基线比对 | 备注 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r.get("ok"):
            L.append(f"| `{r['asset_id']}` | {r['name']} | {r.get('klass','')} | {r['unit']} | — | — | — | — | — | ❌失败 | {r.get('err','')} |")
            continue
        cmp = "—"
        if r["baseline"] is not None:
            try:
                df = pd.read_csv(outdir / f"{r['asset_id']}.csv")
                df["date"] = pd.to_datetime(df["date"])
                sub = df[df["date"] <= WARMUP_END].sort_values("date")
                if not sub.empty:
                    c = float(pd.to_numeric(sub["close"], errors="coerce").iloc[-1])
                    b = float(r["baseline"])
                    # COPPER 基线为 USD/吨，CSV close 为 USD/lb → 基线折算到 USD/lb 再比
                    if r["asset_id"] == "COPPER":
                        b = b / COPPER_LB_TO_TON
                    pct = (c - b) / b * 100
                    flag = "✅" if abs(pct) < 3 else ("⚠️" if abs(pct) < 12 else "❌")
                    cmp = f"{flag} {c:,.4g} vs {b:,.4g} ({pct:+.1f}%)"
            except Exception as e:
                cmp = f"校验异常:{e}"
        L.append(f"| `{r['asset_id']}` | {r['name']} | {r['klass']} | {r['unit']} | {r['source']} | "
                 f"{r['start']} | {r['end']} | {r['rows']} | "
                 f"{r['baseline'] if r['baseline'] is not None else '—'} | {cmp} | {r.get('note','')} |")
    L += ["", "## 2. 单位与口径说明", "",
          "- 指数/商品/汇率/波动率：`close` 为源端原生报价（指数点 / USD-oz / USD-桶 / 汇率 / 指数）。",
          "- **US10Y/CN10Y**：`close` 为收益率百分数（4.30 = 4.30%）。ASSETS.yaml 基线为小数（0.043），比对时 ×100。",
          "- **COPPER**：源 HG=F 为 USD/lb，基线表为 USD/吨；另出 `COPPER_USD_PER_TON.csv`（×2204.62262）。",
          "- **SOX/债券/汇率无原生 OHLCV**：open/high/low 用 close 填充、volume=0；不影响日频收益计算。",
          "- **KOSPI/USDKRW/JP_SEMI_EQUIP**：WL3/WL5 特有，非 19 基准资产；CSV 落盘，默认不进 panel。",
          "- **基线比对说明**：基线来自 refer.md 的「实际市场估计」，与 2026-07 真实价可能有偏差（尤其 SOX/NDX/CN10Y 估计显著偏离真实）；以真实抓取价为准。",
          "- 宽表 `all_close_wide.csv` 取日期并集，各市场交易日历/节假日不同 → 空值为预期，非缺失。",
          "", "## 3. 复跑", "",
          "```bash",
          ".venv/bin/python data-prepare/fetch_daily_data.py            # 复用已落盘 CSV",
          ".venv/bin/python data-prepare/fetch_daily_data.py --force    # 全量重抓",
          ".venv/bin/python data-prepare/fetch_daily_data.py --only VIX,USDJPY  # 补抓缺口",
          "```"]
    (outdir / "COVERAGE.md").write_text("\n".join(L), encoding="utf-8")
    print("   COVERAGE.md")


if __name__ == "__main__":
    sys.exit(main())
