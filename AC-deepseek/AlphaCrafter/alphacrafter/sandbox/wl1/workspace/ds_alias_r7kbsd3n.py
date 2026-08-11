python - <<'EOF'
p = 'scripts/miner1_20260716_load_lib_signals2.py'
s = open(p).read()
s = s.replace('path = "factors/" + base if base in lst else "factors/" + max(lst)',
              'path = base if base in lst else max(lst)')
open(p,'w').write(s)
print("patched")
EOF