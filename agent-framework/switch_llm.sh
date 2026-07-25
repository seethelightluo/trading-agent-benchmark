#!/usr/bin/env bash
# switch_llm.sh [glm|gpt] — 在两套 LLM profile 间切换（端点+密钥+模型），互不覆盖、互不丢失。
#
#   glm = 智谱 BigModel  glm-5.2        (https://open.bigmodel.cn/api/coding/paas/v4)
#   gpt = 自建 OpenAI 兼容 gpt-5.6-terra (http://119.45.37.96:8317/v1)
#
# 做三件事：
#   1. AlphaCrafter/.env.<profile>  →  AlphaCrafter/.env   （激活的凭证，run_all.sh / load_dotenv 读这个）
#   2. 把模型名写回 AC config.yaml（miner/screener/trader 三处 model.code）
#   3. 把模型名写回 FM fm_live.yaml（model）
# 两个 profile 文件(.env.glm / .env.gpt) 永久保留；models.json 两套模型条目都在。
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-}"
case "$PROFILE" in
  glm|gpt) ;;
  *) echo "用法: $0 [glm|gpt]"; \
     echo "  glm = 智谱 glm-5.2 (open.bigmodel.cn)"; \
     echo "  gpt = 自建 gpt-5.6-terra (119.45.37.96:8317)"; \
     exit 1;;
esac
ENV_PROFILE="AlphaCrafter/.env.$PROFILE"
[ -f "$ENV_PROFILE" ] || { echo "❌ 缺 $ENV_PROFILE"; exit 1; }
cp "$ENV_PROFILE" AlphaCrafter/.env
MODEL=$(grep -E '^# MODEL:' "$ENV_PROFILE" | head -1 | sed -E 's/^# MODEL:[[:space:]]*//')
[ -n "$MODEL" ] || { echo "❌ $ENV_PROFILE 缺 '# MODEL:' 行"; exit 1; }
python3 - "$MODEL" <<'PY'
import re, sys
model = sys.argv[1]
cfg = "AlphaCrafter/alphacrafter/config.yaml"
t = open(cfg, encoding="utf-8").read()
# 三处 model.code：吃掉引号后的空白与旧注释，统一写新值 + 标准注释（避免重复切换时注释堆叠）
t, n = re.subn(r'(\n[ \t]*code:\s*)"[^"]*"[ \t]*(?:#.*)?',
               lambda m: f'{m.group(1)}"{model}"  # 由 switch_llm.sh 管理',
               t)
open(cfg, "w", encoding="utf-8").write(t)
fm = "FactorMiner/factorminer/configs/fm_live.yaml"
t = open(fm, encoding="utf-8").read()
t = re.sub(r'(\n[ \t]*model:\s*)"[^"]*"[ \t]*(?:#.*)?',
           lambda m: f'{m.group(1)}"{model}"  # 由 switch_llm.sh 管理',
           t)
open(fm, "w", encoding="utf-8").write(t)
print(f"  config.yaml 替换 {n} 处 model.code；fm_live.yaml model 已更新。")
PY
echo "✅ 已切换到 [$PROFILE]  model=$MODEL"
echo "   endpoint = $(grep -E '^OPENAI_API_URL=' AlphaCrafter/.env | cut -d= -f2-)"
echo "   key_len  = $(grep -E '^OPENAI_API_KEY=' AlphaCrafter/.env | cut -d= -f2- | tr -d '\n' | wc -c)"
