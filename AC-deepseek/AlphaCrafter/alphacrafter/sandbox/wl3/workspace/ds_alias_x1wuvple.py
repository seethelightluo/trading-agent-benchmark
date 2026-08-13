
# Verify frozen weight: 14669.11/1457823.16
print(14669.11/1457823.16)
# check if any order history or execution trace available
import glob
print([f for f in glob.glob('*.json')][:20])
