from subprocess import *
import sys,os,signal
from time import sleep
from operator import itemgetter

# Histrank test statistic cutoff for viewing images
thresh = 100

if len(sys.argv) == 3:
    iistart = int(sys.argv[2])
    filedir = sys.argv[1]
elif len(sys.argv) == 2:
    iistart = 1
    filedir = sys.argv[1]
else:
    print "Usage: python pager.py <histrank dir> [histrank start line]"

f = open(filedir+'/histrank-all.txt')
lines = f.read()
lines = lines.split('\n')
f.close()

ii = iistart
for line in lines[iistart-1:]:
    line = line.split()

    if len(line) > 0:
        basename = line[4]
        ts = float(line[3])
        print ii,basename,ts

        if ts > thresh:
            pr = Popen('exec eog '+filedir+'/*'+basename+'*.png',shell=True,stdin=PIPE)
            sleep(3)
            #raw_input()
            pr.terminate()
    ii = ii+1
