python - <<'EOF'
p = 'scripts/trader_20260730_probe.py'
s = open(p).read()
s = s.replace("down_mask = spx_ret < 0", "down_mask = (spx_ret < 0).reindex(panel.index)")
open(p,'w').write(s)
print('patched')
EOF