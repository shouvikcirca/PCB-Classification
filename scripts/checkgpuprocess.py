import subprocess
import os

count = 1
while True:
    a = subprocess.run('nvidia-smi --query -d pids|grep -n "Process ID"|cut -f 3 -d ":"', capture_output = True, shell = True)
    rp = a.stdout.decode().split('\n')
    if !rp:
        os.system('python3 cutmix_keras'+str(count)+'.py')
        print('Done executing cutmix_keras'+str(count)+'.py')
        break

    
