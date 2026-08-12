
lines = open('scripts/factor_research_lib.py').read().split('\n') if __import__('os').path.exists('scripts/factor_research_lib.py') else []
print(f"exists: {bool(lines)}")
print('\n'.join(lines[:80]) if lines else "no lib file")
