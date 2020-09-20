def entries():
    with open('IVS.xml','r') as f:
        s = f.read()

    s = s.split('\n')
    s = s[4:-3]
    s = [i[5:-3] for i in s]
    s = [i.split(" ")[1:] for i in s]

    for i in range(len(s)):
        s[i][0] = s[i][0].split("=")[1]
        s[i][0] = s[i][0][1:-1]
        s[i][1] = s[i][1].split("=")[1]
        s[i][1] = s[i][1][1:-1]
        s[i][2] = s[i][2].split("=")[1]
        s[i][2] = s[i][2][1:-1]

    return s
