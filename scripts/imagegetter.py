import matplotlib.pyplot as plt
import xmlparser
from PIL import Image

ents = xmlparser.entries()
validcount = 0
invalidcount = 0

for i in range(len(ents)):
    a = ents[i][1].split('\\')[4:]
    a = '/'.join(a)
    try:
        img = Image.open(a)
        validcount+=1
    except:
        invalidcount+=1

print('{} valid images'.format(validcount))
print('{} invalid images'.format(invalidcount))
