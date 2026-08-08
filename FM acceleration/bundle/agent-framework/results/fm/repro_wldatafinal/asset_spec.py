"""
asset_spec.py — 资产单一事实源（asset_id ↔ 数据源候选列表/单位/基线）

被 fetch_daily_data.py 与 make_panel.py 共享，避免映射口径漂移。

== 数据源可靠性结论（2026-07 实测，本机走 Clash TUN）==
  sina / 163（国内）       : 稳定 ✅ —— A 股/港股/美股指数、环球指数(限 ~1000 行)
  eastmoney _em（国内）    : RemoteDisconnected 频发，需重试，间歇可用 ⚠️ —— 商品期货/外汇/DXY/环球指数全量
  Yahoo / FRED（国际）     : 出口 IP 被 rate-limit（429 / timeout）❌ —— 仅作长尾兜底
  Binance（国际）          : 稳定 ✅ —— 加密
  akshare bond/macro（国内): 稳定 ✅ —— 中美国债收益率、SOX

故每个 asset 给出 sources 优先级列表，fetcher 依序尝试，先成先用；eastmoney 类自带重试。

fetcher key 对照（见 fetch_daily_data.py 的 SOURCES）：
  sina_zh, sina_us, sina_hk, sina_global, em_global, em_futures, em_forex,
  sox, us10y, cn10y, binance, yahoo, cboe_vix, proxy_jp_semi
"""
from __future__ import annotations

ASSET_SPEC: list[dict] = [
    # ---- A 股指数（sina/163）----
    dict(asset_id="000300.SH", name="沪深300", klass="equity", unit="指数点",
         baseline=4608, note="朝阳永续权威",
         sources=[("sina_zh", "sh000300")]),
    dict(asset_id="000688.SH", name="科创50", klass="equity", unit="指数点",
         baseline=1920, note="实际市场(2026-07-16)",
         sources=[("sina_zh", "sh000688")]),
    # ---- 美股指数（sina）----
    dict(asset_id="SPX", name="标普500", klass="equity", unit="指数点",
         baseline=7534, note="朝阳永续权威",
         sources=[("sina_us", ".INX")]),
    dict(asset_id="NDX", name="纳斯达克100", klass="equity", unit="指数点",
         baseline=20500, note="实际市场估计",
         sources=[("sina_us", ".NDX")]),
    dict(asset_id="SOX", name="费城半导体", klass="equity", unit="指数点",
         baseline=5800, note="实际市场估计（基线偏低，2026-07 实际~11700）",
         sources=[("sox", None)]),
    # ---- 港股指数（sina）----
    dict(asset_id="HSI", name="恒生指数", klass="equity", unit="指数点",
         baseline=24586, note="朝阳永续权威",
         sources=[("sina_hk", "HSI")]),
    # ---- 环球指数（yahoo 全量优先 → em → sina ~1000行兜底；yahoo 需 US/JP 节点）----
    dict(asset_id="N225", name="日经225", klass="equity", unit="指数点",
         baseline=68000, note="实际市场取整",
         sources=[("yahoo", "^N225"), ("em_global", "日经225指数"), ("sina_global", "日经225指数")]),
    dict(asset_id="SX5E", name="欧洲斯托克50", klass="equity", unit="指数点",
         baseline=5100, note="",
         sources=[("yahoo", "^STOXX50E"), ("em_global", "欧洲Stoxx50指数"), ("sina_global", "欧洲Stoxx50指数")]),
    dict(asset_id="KOSPI", name="韩国KOSPI", klass="equity", unit="指数点",
         baseline=None, note="WL3 特有，非基准资产",
         sources=[("yahoo", "^KS11"), ("sina_global", "首尔综合指数")]),
    # ---- 商品期货（sina foreign futures 优先，稳定；铜 sina 报价 cents/lb → scale 0.01）----
    dict(asset_id="XAU", name="黄金", klass="commodity", unit="USD/oz",
         baseline=4050, note="COMEX 黄金期货≈XAU",
         sources=[("sina_foreign", "GC"), ("em_futures", "GC00Y"), ("yahoo", "GC=F")]),
    dict(asset_id="COPPER", name="铜", klass="commodity", unit="USD/lb",
         baseline=13600, note="基线为 USD/吨；本资产 USD/lb，附 USD/吨列(×2204.62262)",
         sources=[("sina_foreign", "HG", 0.01), ("em_futures", "HG00Y"), ("yahoo", "HG=F")]),
    dict(asset_id="WTI", name="WTI原油", klass="commodity", unit="USD/桶",
         baseline=79, note="",
         sources=[("sina_foreign", "CL"), ("em_futures", "CL00Y"), ("yahoo", "CL=F")]),
    # ---- 加密（binance）----
    dict(asset_id="BTC", name="比特币", klass="crypto", unit="USD(USDT)",
         baseline=64800, note="klines 日线",
         sources=[("binance", "BTCUSDT"), ("yahoo", "BTC-USD")]),
    dict(asset_id="ETH", name="以太坊", klass="crypto", unit="USD(USDT)",
         baseline=1920, note="klines 日线",
         sources=[("binance", "ETHUSDT"), ("yahoo", "ETH-USD")]),
    # ---- 国债收益率 ----
    dict(asset_id="US10Y", name="美债10Y收益率", klass="bond", unit="%(如4.30=4.30%)",
         baseline=4.30, note="close=收益率%; 基线0.043→4.30; akshare bond_zh_us_rate",
         sources=[("us10y", None), ("yahoo", "^TNX")]),
    dict(asset_id="CN10Y", name="中债10Y收益率", klass="bond", unit="%(如1.74=1.74%)",
         baseline=2.20, note="close=收益率%; 中债国债收益率曲线 10年列; 基线0.022→2.20(估计偏高, 实际~1.74)",
         sources=[("cn10y", None)]),
    # ---- 汇率/美元指数（BOC 央行中间价 国内优先 → yahoo 兜底）----
    dict(asset_id="DXY", name="美元指数", klass="fx", unit="指数",
         baseline=100.5, note="BOC 6 成分篮子公式合成（国内，免 Yahoo）",
         sources=[("boc_dxy", None), ("em_global", "美元指数"), ("yahoo", "DX-Y.NYB")]),
    dict(asset_id="USDCNY", name="美元/人民币", klass="fx", unit="汇率",
         baseline=6.78, note="BOC 美元央行中间价/100",
         sources=[("boc_usdcny", None), ("em_forex", "USDCNH"), ("yahoo", "CNY=X")]),
    dict(asset_id="USDJPY", name="美元/日元", klass="fx", unit="汇率",
         baseline=162, note="WL5 特有；BOC 美元/日元交叉；用于日经225 USD 折算",
         sources=[("boc_usdjpy", None), ("em_forex", "USDJPY"), ("yahoo", "JPY=X")]),
    dict(asset_id="EURUSD", name="欧元/美元", klass="fx", unit="汇率",
         baseline=1.08, note="BOC 欧元/美元交叉；用于斯托克50 USD 折算（P_usd=P×EURUSD）",
         sources=[("boc_eur", None), ("em_forex", "EURUSD"), ("yahoo", "EURUSD=X")]),
    dict(asset_id="USDKRW", name="美元/韩元", klass="fx", unit="汇率",
         baseline=None, note="WL3 特有，非基准资产；BOC 美元/韩国元交叉",
         sources=[("boc_usdkrw", None), ("em_forex", "USDKRW"), ("yahoo", "KRW=X")]),
    # ---- 波动率 ----
    dict(asset_id="VIX", name="波动率指数", klass="vol", unit="指数",
         baseline=16, note="yahoo/cboe 优先；出口 IP 可能 429，失败则作缺口待补",
         sources=[("yahoo", "^VIX"), ("cboe_vix", None), ("em_global", "VIX")]),
    # ---- 可选代理（WL5）----
    dict(asset_id="JP_SEMI_EQUIP", name="日本半导体设备(代理)", klass="equity", unit="等权归一指数",
         baseline=None, note="WL5 可选; Advantest(6857.T)+Tokyo Electron(8035.T) 等权, 2020-01-02 归一为 1.0; 非基准资产",
         sources=[("proxy_jp_semi", None)]),
]

# === 资产口径（2026-07-22 关键修订，见 plan.md §7）===
# 数据层全量 = 20 = 15 可交易 + 5 信号。信号项（DXY/USDCNY/USDJPY/EURUSD/VIX）仅作 Agent
# 输入特征，不进持仓权重向量 w_t；其中 USDCNY/USDJPY/EURUSD 兼做对应外币资产 USD 折算。

# 可交易持仓（15）：权益8 + 商品3 + 加密2 + 债券2
TRADABLE_ASSET_IDS = [
    "000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
    "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y",
]

# 宏观/状态信号（5）：仅特征，不持仓
SIGNAL_ASSET_IDS = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

# 数据层基准资产（make_panel / gen_worldline_online 默认全量）：20
BENCHMARK_ASSET_IDS = TRADABLE_ASSET_IDS + SIGNAL_ASSET_IDS

# 原生计价货币 → USD 折算（§7.2）。op: div=P/rate, mul=P×rate。rate 为汇率 asset_id 或常数。
# 未列出的（SPX/NDX/SOX/XAU/COPPER/WTI/BTC/ETH/US10Y）原生 USD，无需折算。
# HSI 用 HKD-USD 联系汇率常数 7.80（peg，无需抓取）。
TO_USD = {
    "000300.SH": dict(ccy="cny", rate="USDCNY", op="div"),
    "000688.SH": dict(ccy="cny", rate="USDCNY", op="div"),
    "CN10Y":     dict(ccy="cny", rate="USDCNY", op="div"),
    "HSI":       dict(ccy="hkd", rate=7.80,     op="div"),  # 联系汇率常数
    "N225":      dict(ccy="jpy", rate="USDJPY", op="div"),
    "SX5E":      dict(ccy="eur", rate="EURUSD", op="mul"),
}

# 铜 USD/lb → USD/吨（1 ton = 2204.62262 lb）
COPPER_LB_TO_TON = 2204.62262
