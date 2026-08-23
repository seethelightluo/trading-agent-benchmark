cd /workspace && python scripts/miner2_20270617_revalidate_full.py 2>&1 || python -c "
import sys
sys.path.insert(0, 'scripts')
exec(open('scripts/miner2_20270617_revalidate_full.py').read())
" 2>&1 | head -50