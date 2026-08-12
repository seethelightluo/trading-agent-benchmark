
lines = open('memory.txt').readlines()
for i in range(46, min(60, len(lines))):
    print(i+1, '|', lines[i][:600])
