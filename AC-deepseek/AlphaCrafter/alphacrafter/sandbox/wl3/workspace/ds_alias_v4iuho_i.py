
with open('strategy.py') as f:
    lines = f.readlines()
print("total lines:", len(lines))
for i in range(395, min(470, len(lines))):
    print(f'{i+1}: {lines[i]}', end='')
