import os
print([f for f in os.listdir('scripts') if 'factor_research' in f or 'research' in f.lower()])
print("---")
print(open('scripts/factor_research_lib.py').read()[:5000])