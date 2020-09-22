import matplotlib.pyplot as plt
import xmlparser
from PIL import Image
import os

ents = xmlparser.entries()
validcount = 0
invalidcount = 0

di = {} 
for i in range(len(ents)):
    a = ents[i][1].split('\\')[4:]
    if a[3] not in di:
        di[a[3]] = True
    a = '/'.join(a)
    try:
        b = Image.open(a)
        validcount+=1
    except:
        invalidcount+=1

print('{} valid images'.format(validcount))
print('{} invalid images'.format(invalidcount))

for item in di:
    print(item)

print(type(b))
