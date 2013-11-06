import sys,os
from subprocess import *
from glob import glob
import gc

if len(sys.argv) != 2:
    print 'Usage: spplot.py [topdir]'
    print '(Topdir is the parent of all beam dirs for a given drift strip)'
    sys.exit(1)

topdir = sys.argv[1]

if not os.path.exists(topdir):
    print 'Directory '+topdir+' does not exist!'
    sys.exit(1)

os.chdir(topdir)
dirs = glob('D*')
dirs.sort()

n = len(dirs)
ii = 1
for d in dirs:
    print '*** Dir %d of %d ***' %(ii,n)
    # Have to do this to go around (pylab?) memory leaks    
    pr = Popen('python /home/deneva/drift/spplot.py '+d,shell=True,stdin=PIPE)
    pr.wait()
    del pr

    ii = ii+1



