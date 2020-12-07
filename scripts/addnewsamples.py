import os

olddir = 'old_fold1/'
newdir = 'new_fold1/'

oftraintrues = os.listdir(olddir + 'Train/True')
#nftraintrues = os.listdir(newdir + 'Train/True')

oftrainfalses = os.listdir(olddir + 'Train/False')
#nftrainfalses = os.listdir(newdir + 'Train/False')

ofvaltrues = os.listdir(olddir + 'Validation/True')
#nfvaltrues = os.listdir(newdir + 'Validation/True')

ofvalfalses = os.listdir(olddir + 'Validation/False')
#nfvalfalses = os.listdir(newdir + 'Validation/False')

balancedtesttrues = os.listdir('BalancedTest/True')
balancedtestfalses = os.listdir('BalancedTest/False')


"""
#Checking for overlap in train, validation and test sets
for i in oftraintrues:
    if i in ofvaltrues or i in balancedtesttrues:
        print('overlap')

for i in ofvaltrues:
    if i in balancedtesttrues:
        print('overlap')


for i in oftrainfalses:
    if i in ofvalfalses or i in balancedtestfalses:
        print('overlap')


for i in ofvaltrues:
    if i in balancedtestfalses:
        print('overlap')

"""

addedtrues = []
addedfalses = []

for im in nftraintrues:
    if im not in oftraintrues and im not in balancedtesttrues and im not in ofvaltrues:
        addedtrues.append(newdir + 'Train/True/' + im)

for im in nftrainfalses:
    if im not in oftrainfalses and im not in balancedtestfalses and im not in ofvalfalses:
        addedfalses.append(newdir +'Train/False/' +im)

for im in nfvaltrues:
    if im not in ofvaltrues and im not in balancedtesttrues and im not in oftraintrues:
        addedtrues.append(newdir + 'Validation/True/' +im)

for im in nfvalfalses:
    if im not in ofvalfalses and im not in balancedtestfalses and im not in oftrainfalses:
        addedfalses.append(newdir + 'Validation/False/' +im)

print(len(addedtrues))
print(len(addedfalses))

newtruetraincount = int(0.6*len(addedtrues))

for i in addedtrues[:newtruetraincount]:
    execstr  = 'cp '+ i+ ' old_fold3/Train/True'
    print(execstr)
    os.system(execstr)


for i in addedfalses[:newtruetraincount]:
    execstr  = 'cp '+ i+ ' old_fold3/Train/False'
    print(execstr)
    os.system(execstr)

"""
